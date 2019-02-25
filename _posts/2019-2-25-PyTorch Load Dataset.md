---
layout:     post                    # 使用的布局（不需要改）
title:      PyTorch——Load Dataset         # 标题
date:       2019-2-25             # 时间
author:     Kiri                      # 作者
header-img: img/post-bg-img.jpg   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 笔记
---
# PyTorch Load Dataset

```python
import torch

import numpy as np
import cv2

import torch.utils.data as data

import os
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.io as scio
from torchvision import transforms


def bbox_2d_coordinate(data, bias):
    """
    计算2D bbox的坐标
    :param data: 单幅图片中Joints的坐标
    :param bias: 偏差
    :return: 2D bbox四个顶点坐标
    """

    u_max = data[:, 0].max()
    u_min = data[:, 0].min()

    v_max = data[:, 1].max()
    v_min = data[:, 1].min()

    width = u_max - u_min + bias
    height = v_max - v_min + bias

    middle_pt_u = (u_min + u_max) * 0.5
    middle_pt_v = (v_min + v_max) * 0.5

    u1 = middle_pt_u - 0.5 * width
    u2 = middle_pt_u + 0.5 * width

    v1 = middle_pt_v - 0.5 * height
    v2 = middle_pt_v + 0.5 * height

    bbox = np.array([u1, v1, width, height])

    return bbox

class NYUDataset(data.Dataset):

    def __init__(self, root_dir, mat_file, transfrom=None):
        # 初始化数据集根目录
        self.root_dir = root_dir
        # 初始化标注文件的目录
        self.mat_file_root = os.path.join(self.root_dir, mat_file)
        # 加载标注文件内的数据
        self.joint = scio.loadmat(self.mat_file_root)
        # 读取像素坐标
        self.joint_uvd = self.joint['joint_uvd']
        # 读取空间坐标
        self.joint_xyz = self.joint['joint_xyz']
        self.transform = transfrom

    def __len__(self):
        return self.joint_uvd[0].shape[0]

    def __getitem__(self, idx):
        img_name = 'rgb_1_{:07}.png'.format(idx + 1)
        file_root = os.path.join(self.root_dir, img_name)
        img = io.imread(file_root)
        images = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        labels = np.ones((1, 1))
        joints = self.joint_uvd[0, idx]
        bbox = bbox_2d_coordinate(joints, 20)
        samples = {'images': images, 'labels': labels, 'joints': joints, 'bbox': bbox, 'img': img}

        if self.transform:
            samples = self.transform(samples)

        return samples


class ToTensor(object):
    # 将数据转换成Tensor

    def __call__(self, sample):
        images, labels, joints, bbox, img = sample['images'], sample['labels'], sample['joints'], sample['bbox'], sample['img']

        images = images.transpose((2, 0, 1))
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels),
                'joints': torch.from_numpy(joints),
                'bbox': torch.from_numpy(bbox),
                'img': img}
```

