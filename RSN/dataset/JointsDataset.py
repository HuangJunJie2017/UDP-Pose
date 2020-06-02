"""
@author: Yuanhao Cai
@date:  2020.03
"""

import copy
import cv2
import numpy as np
import random

import torch
import math
from torch.utils.data import Dataset

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from lib.utils.transforms import flip_joints

kernelsize2sigma ={
    15:2.6,
    11:2.0,
    9:1.7,
    7:1.4105,
    5:1.1105,
}

def get_warpmatrix_inverse(theta, size_input, size_dst, size_target):
    """

    :param theta: angle x y
    :param size_input:[w,h]
    :param size_dst: [w,h] i
    :param size_target: [w,h] b
    :return:
    """
    size_target = size_target * 200.0
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0]/size_target[0]
    scale_y = size_dst[1]/size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x*(-0.5*size_input[0]*math.cos(theta) + 0.5*size_input[1]*math.sin(theta) + 0.5*size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y*(-0.5*size_input[0]*math.sin(theta) - 0.5*size_input[1]*math.cos(theta) + 0.5*size_target[1])
    return matrix

def get_warpmatrix(theta,size_input,size_dst,size_target):
    '''

    :param theta: angle
    :param size_input:[w,h]
    :param size_dst: [w,h]
    :param size_target: [w,h]/200.0
    :return:
    '''
    size_target = size_target * 200.0
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2,3),dtype=np.float32)
    scale_x = size_target[0]/size_dst[0]
    scale_y = size_target[1]/size_dst[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = math.sin(theta) * scale_y
    matrix[0, 2] = -0.5 * size_target[0] * math.cos(theta) - 0.5 * size_target[1] * math.sin(theta) + 0.5 * size_input[0]
    matrix[1, 0] = -math.sin(theta) * scale_x
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = 0.5*size_target[0]*math.sin(theta)-0.5*size_target[1]*math.cos(theta)+0.5*size_input[1]
    return matrix

class JointsDataset(Dataset):

    def __init__(self, DATASET, stage, transform=None):
        self.stage = stage
        assert self.stage in ('train', 'val', 'test')

        self.transform = transform
        self.data = list()

        self.keypoint_num = DATASET.KEYPOINT.NUM
        self.flip_pairs = DATASET.KEYPOINT.FLIP_PAIRS
        self.upper_body_ids = DATASET.KEYPOINT.UPPER_BODY_IDS
        self.lower_body_ids = DATASET.KEYPOINT.LOWER_BODY_IDS
        self.kp_load_min_num = DATASET.KEYPOINT.LOAD_MIN_NUM

        self.input_shape = DATASET.INPUT_SHAPE
        self.output_shape = DATASET.OUTPUT_SHAPE
        self.w_h_ratio = DATASET.WIDTH_HEIGHT_RATIO 

        self.pixel_std = DATASET.PIXEL_STD
        self.color_rgb = DATASET.COLOR_RGB

        self.basic_ext = DATASET.TRAIN.BASIC_EXTENTION
        self.rand_ext = DATASET.TRAIN.RANDOM_EXTENTION
        self.x_ext = DATASET.TRAIN.X_EXTENTION
        self.y_ext = DATASET.TRAIN.Y_EXTENTION
        self.scale_factor_low = DATASET.TRAIN.SCALE_FACTOR_LOW
        self.scale_factor_high = DATASET.TRAIN.SCALE_FACTOR_HIGH
        self.scale_shrink_ratio = DATASET.TRAIN.SCALE_SHRINK_RATIO
        self.rotation_factor = DATASET.TRAIN.ROTATION_FACTOR
        self.prob_rotation = DATASET.TRAIN.PROB_ROTATION
        self.prob_flip = DATASET.TRAIN.PROB_FLIP
        self.num_keypoints_half_body = DATASET.TRAIN.NUM_KEYPOINTS_HALF_BODY
        self.prob_half_body = DATASET.TRAIN.PROB_HALF_BODY
        self.x_ext_half_body = DATASET.TRAIN.X_EXTENTION_HALF_BODY
        self.y_ext_half_body = DATASET.TRAIN.Y_EXTENTION_HALF_BODY
        self.add_more_aug = DATASET.TRAIN.ADD_MORE_AUG
        self.gaussian_kernels = DATASET.TRAIN.GAUSSIAN_KERNELS

        self.test_x_ext = DATASET.TEST.X_EXTENTION
        self.test_y_ext = DATASET.TEST.Y_EXTENTION

        feat_width = self.output_shape[1]
        feat_height = self.output_shape[0]
        feat_x_int = np.arange(0, feat_width)
        feat_y_int = np.arange(0, feat_height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        self.feat_x_int = feat_x_int.reshape((-1,))
        self.feat_y_int = feat_y_int.reshape((-1,))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        d = copy.deepcopy(self.data[idx])

        img_id = d['img_id']
        img_path = d['img_path']

        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if data_numpy is None:
            raise ValueError('fail to read {}'.format(img_path))

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = d['joints'][:, :2]
        joints_vis = d['joints'][:, -1].reshape((-1, 1))
        
        center = d['center']
        scale = d['scale']
        score = d['score'] if 'score' in d else 1
        rotation = 0

        if self.stage == 'train':
            scale[0] *= (1 + self.basic_ext)
            scale[1] *= (1 + self.basic_ext)
            rand = np.random.rand() if self.rand_ext else 1.0
            scale[0] *= (1 + rand * self.x_ext)
            rand = np.random.rand() if self.rand_ext else 1.0
            scale[1] *= (1 + rand * self.y_ext)
        else:
            scale[0] *= (1 + self.test_x_ext)
            scale[1] *= (1 + self.test_y_ext)

        # fit the ratio
        if scale[0] > self.w_h_ratio * scale[1]:
            scale[1] = scale[0] * 1.0 / self.w_h_ratio
        else:
            scale[0] = scale[1] * 1.0 * self.w_h_ratio

        # augmentation
        if self.stage == 'train':
            # half body
            if (np.sum(joints_vis[:, 0] > 0) > self.num_keypoints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis)

                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            # scale
            rand = random.uniform(
                    1 + self.scale_factor_low, 1 + self.scale_factor_high)
            scale_ratio = self.scale_shrink_ratio * rand
            scale *= scale_ratio

            # rotation
            if random.random() <= self.prob_rotation:
                rotation = random.uniform(
                        -self.rotation_factor, self.rotation_factor)

            # flip
            if random.random() <= self.prob_flip:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = flip_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1


        trans = get_warpmatrix(rotation, center * 2.0,
                               np.array([self.input_shape[1], self.input_shape[0]], dtype=float) - 1.0, scale)
        trans_inv = get_warpmatrix_inverse(rotation, center * 2.0,
                               np.array([self.input_shape[1], self.input_shape[0]], dtype=float) - 1.0, scale)
        img = cv2.warpAffine(data_numpy, trans, (int(self.input_shape[1]), int(self.input_shape[0])),
                             flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        if self.transform:
            img = self.transform(img)

        if self.stage == 'train':
            for i in range(self.keypoint_num):
                if joints_vis[i, 0] > 0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans_inv)
                    if joints[i, 0] < 0 \
                            or joints[i, 0] > self.input_shape[1] - 1 \
                            or joints[i, 1] < 0 \
                            or joints[i, 1] > self.input_shape[0] - 1:
                        joints_vis[i, 0] = 0
            valid = torch.from_numpy(joints_vis).float()

            labels_num = len(self.gaussian_kernels)
            labels = np.zeros(
                    (labels_num, self.keypoint_num, *self.output_shape))
            for i in range(labels_num):
                labels[i] = self.generate_heatmap(
                        joints, valid, kernel=self.gaussian_kernels[i])
            labels = torch.from_numpy(labels).float()

            return img, valid, labels
        else:
            return img, score, center, scale, img_id

    def _get_data(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.keypoint_num):
            if joints_vis[joint_id, 0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 3:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 3 else upper_joints

        if len(selected_joints) < 3:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        center = (left_top + right_bottom) / 2

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        rand = np.random.rand()
        w *= (1 + rand * self.x_ext_half_body)
        rand = np.random.rand()
        h *= (1 + rand * self.y_ext_half_body)

        if w > self.w_h_ratio * h:
            h = w * 1.0 / self.w_h_ratio
        elif w < self.w_h_ratio * h:
            w = h * self.w_h_ratio

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)

        return center, scale

    def generate_heatmap(self, joints, valid, kernel=(7, 7)):
        heatmaps = np.zeros(
                (self.keypoint_num, self.output_shape[0]*self.output_shape[1]), dtype='float32')

        for i in range(self.keypoint_num):
            if valid[i] < 1:
                continue
            target_y = joints[i, 1] * (self.output_shape[0]-1.0) / (self.input_shape[0]-1.0)
            target_x = joints[i, 0] * (self.output_shape[1] -1.0) / (self.input_shape[1]-1.0)
            x_offset = (target_x - self.feat_x_int)
            y_offset = (target_y - self.feat_y_int)
            dis = x_offset ** 2 + y_offset ** 2
            sigma = kernelsize2sigma[kernel[0]]
            heatmaps[i] = 255.0 * np.exp(dis / (-2.0 * sigma ** 2))
        heatmaps = heatmaps.reshape((self.keypoint_num, *self.output_shape))
        return heatmaps
