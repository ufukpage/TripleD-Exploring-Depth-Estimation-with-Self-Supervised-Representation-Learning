#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import os
from .utils import readlines
from torchvision.datasets import Cityscapes
from mono.datasets.kitti_dataset import KittiSegmentation
from .labels_file import labels_cityscape_seg
from .mytransforms import *


def get_test_segmentation_dataset(cfg, val=True):
    dataset_name = cfg['name']
    labels = labels_cityscape_seg.getlabels()
    split_str = 'val' if val else 'test'
    if dataset_name == 'kitti':
        kitti_transforms = Compose([
                Resize((cfg.height, cfg.width), only_img=True),
                ConvertSegmentation(labels=labels),
                ToTensor(),
                NormalizeZeroMean()
        ])
        dataset = KittiSegmentation(cfg.in_path, split=split_str, transform=kitti_transforms)
    elif dataset_name == 'cityscapes':
        cs_transforms = Compose([
                Resize((cfg.height, cfg.width), only_img=True ),
                ConvertSegmentation(labels=labels),
                ToTensor(),
                NormalizeZeroMean()
        ])
        dataset = Cityscapes(cfg.in_path, split=split_str, mode='fine', target_type='semantic',
                             transforms=cs_transforms)
    dataset.flag = np.zeros(dataset.__len__(), dtype=np.int64) # workaround
    return dataset


def get_segmentation_train_dataset(cfg, training=True):
    dataset_name = cfg['name']


    labels = labels_cityscape_seg.getlabels()

    if dataset_name == 'kitti':
        kitti_transforms = Compose([
                RandomHorizontalFlip(0.5),
                Resize((cfg.height, cfg.width)),
                # RandomRescale(1.5),
                # RandomCrop((cfg.height, cfg.width)),
                ConvertSegmentation(labels=labels),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
                ToTensor(),
                NormalizeZeroMean()
        ])
        dataset = KittiSegmentation(cfg.in_path, split='train', transform=kitti_transforms)
    elif dataset_name == 'cityscapes':
        cs_transforms = Compose([
                RandomHorizontalFlip(0.5),
                Resize((512, 1024)),
                RandomRescale(1.5),
                RandomCrop((cfg.height, cfg.width)),
                ConvertSegmentation(labels=labels),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.2),
                ToTensor(),
                NormalizeZeroMean()
        ])
        dataset = Cityscapes(cfg.in_path, split='train', mode='fine', target_type='semantic',
                             transforms=cs_transforms)
    dataset.flag = np.zeros(dataset.__len__(), dtype=np.int64) # workaround
    return dataset


def get_dataset(cfg, training=True):
    dataset_name = cfg['name']
    if dataset_name == 'kitti':
        from .kitti_dataset import KITTIRAWDataset as dataset
    if dataset_name == 'kitti_map':
        from .kitti_dataset import KITTIMAPDataset as dataset
    elif dataset_name == 'kitti_inpaint':
        from .kitti_dataset import KITTIInpaintDataset as dataset
    elif dataset_name == 'kitti_odom':
        from .kitti_dataset import KITTIOdomDataset as dataset
    elif dataset_name == 'cityscape':
        from .cityscape_dataset import CityscapeDataset as dataset
    elif dataset_name == 'folder':
        from .folder_dataset import FolderDataset as dataset
    elif dataset_name == 'eth3d':
        from .eth3d_dataset import FolderDataset as dataset
    elif dataset_name == 'euroc':
        from .euroc_dataset import FolderDataset as dataset

    fpath = os.path.join(os.path.dirname(__file__), "splits", cfg.split, "{}_files.txt")
    filenames = readlines(fpath.format("train")) if training else readlines(fpath.format('val'))
    img_ext = '.png' if cfg.png == True else '.jpg'

    dataset = dataset(cfg.in_path,
                      filenames,
                      cfg.height,
                      cfg.width,
                      cfg.frame_ids if training else [0],
                      is_train=training,
                      img_ext=img_ext,
                      gt_depth_path=cfg.gt_depth_path, cfg=cfg)
    return dataset