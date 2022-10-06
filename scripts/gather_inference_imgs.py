from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
MIN_DEPTH=1e-3
MAX_DEPTH=80

if __name__ == "__main__":
    dest_path = "./gather_imgs"
    os.makedirs(dest_path, exist_ok=True)

    td_path = './train_monodepth2_disentangle_full_hp3'
    md2_path = "./monodepth2_outs"
    fd_path = "./fm_depth"



    # setting values to rows and column variables
    rows = 2
    columns = 2

    for ind in range(697):
        img_path = os.path.join(td_path, 'disp_{:0>4d}.jpg'.format(ind))
        td_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_path = os.path.join(md2_path, 'disp_{:0>4d}.jpg'.format(ind))
        md2_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_path = os.path.join(fd_path, 'disp_{:0>4d}.jpg'.format(ind))
        fd_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_path = os.path.join(fd_path, 'img_{:0>4d}.jpg'.format(ind))
        rgb_img = cv2.imread(img_path)

        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)

        plt.imshow(md2_img, cmap='magma')
        plt.axis('off')
        plt.title("Monodepth2")
        fig.add_subplot(rows, columns, 2)

        plt.imshow(fd_img, cmap='magma')
        plt.axis('off')
        plt.title("Feat Depth")

        fig.add_subplot(rows, columns, 3)
        plt.imshow(td_img, cmap='magma')
        plt.axis('off')
        plt.title("TripleD")

        fig.add_subplot(rows, columns, 4)
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.title("RGB")

        img_path = os.path.join(dest_path, 'all_{:0>4d}.jpg'.format(ind))
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        #plt
        """"
        all = np.concatenate((md2_img, fd_img, td_img, rgb_img), axis=0)
        cv2.imshow('all', all)

        cv2.waitKey(0)
        """