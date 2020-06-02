"""
@author: Yuanhao Cai
@date:  2020.03
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import json

import torch
import torch.distributed as dist

from cvpack.utils.logger import get_logger

from config import cfg
from network import RSN
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process, synchronize, all_gather
from lib.utils.transforms import flip_back

import numpy as np
from numpy.linalg import LinAlgError
# from utils.transforms import transform_preds
import cv2

def transform_preds(coords, center, scale, output_size):
    scale = scale * 200.0
    scale_x = scale[0]/(output_size[0]-1.0)
    scale_y = scale[1]/(output_size[1]-1.0)
    target_coords = np.zeros(coords.shape)
    target_coords[:,0] = coords[:,0]*scale_x + center[0]-scale[0]*0.5
    target_coords[:,1] = coords[:,1]*scale_y + center[1]-scale[1]*0.5
    # trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    # for p in range(coords.shape[0]):
    #     target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def post(coords,batch_heatmaps):
    '''
    DARK post-pocessing
    :param coords: batchsize*num_kps*2
    :param batch_heatmaps:batchsize*num_kps*high*width
    :return:
    '''

    shape_pad = list(batch_heatmaps.shape)
    shape_pad[2] = shape_pad[2] + 2
    shape_pad[3] = shape_pad[3] + 2

    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            mapij=batch_heatmaps[i,j,:,:]
            maxori = np.max(mapij)
            mapij= cv2.GaussianBlur(mapij,(7, 7), 0)
            max = np.max(mapij)
            min = np.min(mapij)
            mapij = (mapij-min)/(max-min) * maxori
            batch_heatmaps[i, j, :, :]= mapij
    batch_heatmaps = np.clip(batch_heatmaps,0.001,50)
    batch_heatmaps = np.log(batch_heatmaps)
    batch_heatmaps_pad = np.zeros(shape_pad,dtype=float)
    batch_heatmaps_pad[:, :, 1:-1,1:-1] = batch_heatmaps
    batch_heatmaps_pad[:, :, 1:-1, -1] = batch_heatmaps[:, :, :,-1]
    batch_heatmaps_pad[:, :, -1, 1:-1] = batch_heatmaps[:, :, -1, :]
    batch_heatmaps_pad[:, :, 1:-1, 0] = batch_heatmaps[:, :, :, 0]
    batch_heatmaps_pad[:, :, 0, 1:-1] = batch_heatmaps[:, :, 0, :]
    batch_heatmaps_pad[:, :, -1, -1] = batch_heatmaps[:, :, -1 , -1]
    batch_heatmaps_pad[:, :, 0, 0] = batch_heatmaps[:, :, 0, 0]
    batch_heatmaps_pad[:, :, 0, -1] = batch_heatmaps[:, :, 0, -1]
    batch_heatmaps_pad[:, :, -1, 0] = batch_heatmaps[:, :, -1, 0]
    I = np.zeros((shape_pad[0],shape_pad[1]))
    Ix1 = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1y1 = np.zeros((shape_pad[0],shape_pad[1]))
    Ix1_y1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1_ = np.zeros((shape_pad[0], shape_pad[1]))
    coords = coords.astype(np.int32)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            I[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0]+1]
            Ix1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0] + 2]
            Ix1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0] ]
            Iy1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0]+1]
            Iy1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] , coords[i, j, 0]+1]
            Ix1y1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0] + 2]
            Ix1_y1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1], coords[i, j, 0]]
    dx = 0.5 * (Ix1 -  Ix1_)
    dy = 0.5 * (Iy1 - Iy1_)
    D = np.zeros((shape_pad[0],shape_pad[1],2))
    D[:,:,0]=dx
    D[:,:,1]=dy
    D.reshape((shape_pad[0],shape_pad[1],2,1))
    dxx = Ix1 - 2*I + Ix1_
    dyy = Iy1 - 2*I + Iy1_
    dxy = 0.5*(Ix1y1- Ix1 -Iy1 + I + I -Ix1_-Iy1_+Ix1_y1_)
    hessian = np.zeros((shape_pad[0],shape_pad[1],2,2))
    hessian[:, :, 0, 0] = dxx
    hessian[:, :, 1, 0] = dxy
    hessian[:, :, 0, 1] = dxy
    hessian[:, :, 1, 1] = dyy
    inv_hessian = np.zeros(hessian.shape)
    # hessian_test = np.zeros(hessian.shape)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            hessian_tmp = hessian[i,j,:,:]
            try:
                inv_hessian[i,j,:,:] = np.linalg.inv(hessian_tmp)
            except LinAlgError:
                inv_hessian[i, j, :, :] = np.zeros((2,2))
            # hessian_test[i,j,:,:] = np.matmul(hessian[i,j,:,:],inv_hessian[i,j,:,:])
            # print( hessian_test[i,j,:,:])
    res = np.zeros(coords.shape)
    coords = coords.astype(np.float)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            D_tmp = D[i,j,:]
            D_tmp = D_tmp[:,np.newaxis]
            shift = np.matmul(inv_hessian[i,j,:,:],D_tmp)
            # print(shift.shape)
            res_tmp = coords[i, j, :] -  shift.reshape((-1))
            res[i,j,:] = res_tmp
    return res


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results = list() 
    cpu_device = torch.device("cpu")

    data = tqdm(data_loader) if is_main_process() else data_loader
    for _, batch in enumerate(data):
        imgs, scores, centers, scales, img_ids = batch

        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.to(cpu_device).numpy()

            if cfg.TEST.FLIP:
                imgs_flipped = np.flip(imgs.to(cpu_device).numpy(), 3).copy()
                imgs_flipped = torch.from_numpy(imgs_flipped).to(device)
                outputs_flipped = model(imgs_flipped)
                outputs_flipped = outputs_flipped.to(cpu_device).numpy()
                outputs_flipped = flip_back(
                        outputs_flipped, cfg.DATASET.KEYPOINT.FLIP_PAIRS)
                outputs = (outputs + outputs_flipped) * 0.5

        centers = np.array(centers)
        scales = np.array(scales)
        outputs = outputs/255.0
        preds, maxvals = get_max_preds(outputs)

        preds = post(preds, outputs)
        for i in range(preds.shape[0]):
            preds[i] = transform_preds(
                preds[i], centers[i], scales[i], [48, 64]
            )


        kp_scores = maxvals.squeeze().mean(axis=1)
        preds = np.concatenate((preds, maxvals), axis=2)

        for i in range(preds.shape[0]):
            keypoints = preds[i].reshape(-1).tolist()
            score = scores[i] * kp_scores[i]
            image_id = img_ids[i]

            results.append(dict(image_id=image_id,
                                category_id=1,
                                keypoints=keypoints,
                                score=score))

    return results


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, logger):
    if is_main_process():
        logger.info("Accumulating ...")
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return

    predictions = list()
    for p in all_predictions:
        predictions.extend(p)
    
    return predictions


def inference(model, data_loader, logger, device="cuda"):
    predictions = compute_on_dataset(model, data_loader, device)
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(
            predictions, logger)

    if not is_main_process():
        return

    return predictions    
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1)
    args = parser.parse_args()

    num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed =  num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if is_main_process() and not os.path.exists(cfg.TEST_DIR):
        os.mkdir(cfg.TEST_DIR)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, args.local_rank, 'test_log.txt')

    if args.iter == -1:
        logger.info("Please designate one iteration.")

    model = RSN(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(cfg.MODEL.DEVICE)

    model_file = os.path.join(cfg.OUTPUT_DIR, "iter-{}.pth".format(args.iter))
    if os.path.exists(model_file):
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    data_loader = get_test_loader(cfg, num_gpus, args.local_rank, 'val',
            is_dist=distributed)

    results = inference(model, data_loader, logger, device)
    synchronize()

    if is_main_process():
        logger.info("Dumping results ...")
        results.sort(
                key=lambda res:(res['image_id'], res['score']), reverse=True) 
        results_path = os.path.join(cfg.TEST_DIR, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        logger.info("Get all results.")

        data_loader.ori_dataset.evaluate(results_path)


if __name__ == '__main__':
    main()
