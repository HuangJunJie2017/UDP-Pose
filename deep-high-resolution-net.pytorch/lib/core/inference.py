# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from numpy.linalg import LinAlgError
from utils.transforms import transform_preds
import cv2

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



def get_final_preds(config, batch_heatmaps, center, scale):
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    if config.MODEL.TARGET_TYPE == 'gaussian':
        coords, maxvals = get_max_preds(batch_heatmaps)
        if config.TEST.POST_PROCESS:
            coords = post(coords,batch_heatmaps)
    elif config.MODEL.TARGET_TYPE == 'offset':
        net_output = batch_heatmaps.copy()
        kps_pos_distance_x = config.LOSS.KPD
        kps_pos_distance_y = config.LOSS.KPD
        batch_heatmaps = net_output[:,::3,:]
        offset_x = net_output[:,1::3,:] * kps_pos_distance_x
        offset_y = net_output[:,2::3,:] * kps_pos_distance_y
        for i in range(batch_heatmaps.shape[0]):
            for j in range(batch_heatmaps.shape[1]):
                batch_heatmaps[i,j,:,:] = cv2.GaussianBlur(batch_heatmaps[i,j,:,:],(15, 15), 0)
                offset_x[i,j,:,:] = cv2.GaussianBlur(offset_x[i,j,:,:],(7, 7), 0)
                offset_y[i,j,:,:] = cv2.GaussianBlur(offset_y[i,j,:,:],(7, 7), 0)
        coords, maxvals = get_max_preds(batch_heatmaps)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                px = int(coords[n][p][0])
                py = int(coords[n][p][1])
                coords[n][p][0] += offset_x[n,p,py,px]
                coords[n][p][1] += offset_y[n,p,py,px]

    preds = coords.copy()
    preds_in_input_space = preds.copy()
    preds_in_input_space[:,:, 0] = preds_in_input_space[:,:, 0] / (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
    preds_in_input_space[:,:, 1] = preds_in_input_space[:,:, 1] / (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals, preds_in_input_space
