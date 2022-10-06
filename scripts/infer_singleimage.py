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


def evaluate(cfg_path,model_path,gt_path, output_path):
    filenames = readlines("./mono/datasets/splits/exp/val_files.txt")
    cfg = Config.fromfile(cfg_path)

    dataset = KITTIRAWDataset(cfg.data['in_path'],
                              filenames,
                              cfg.data['height'],
                              cfg.data['width'],
                              [0],
                              is_train=False,
                              gt_depth_path=gt_path,
                              img_ext='.png' if cfg.data['png'] else '.jpg', cfg=cfg)

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)

    cfg.model['imgs_per_gpu'] = 1
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith('Depth')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    # gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            outputs = model(inputs)

            img_path = os.path.join(output_path, 'img_{:0>4d}.jpg'.format(batch_idx))
            plt.imsave(img_path, inputs[("color", 0, 0)][0].squeeze().transpose(0,1).transpose(1,2).cpu().numpy())

            disp = outputs[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp[0, 0].cpu().numpy()
            pred_disp = cv2.resize(pred_disp, (cfg.data['width'], cfg.data['height']))

            img_path = os.path.join(output_path, 'disp_{:0>4d}.jpg'.format(batch_idx))
            vmax = np.percentile(pred_disp, 95)
            plt.imsave(img_path, pred_disp, cmap='magma', vmax=vmax)

            # gt_depth = gt_depths[batch_idx]
            """"
            gt_depth = inputs["gt_depth"]
            # gt_disp, _ = disp_to_depth(gt_depth, 0.1, 100)
            gt_disp = gt_depth[0, 0].cpu().numpy()
            gt_disp = cv2.resize(gt_disp, (cfg.data['width'], cfg.data['height']))

            img_path = os.path.join(output_path, 'gt_{:0>4d}.jpg'.format(batch_idx))
            gt_disp[gt_disp < MIN_DEPTH] = MIN_DEPTH
            gt_disp[gt_disp > MAX_DEPTH] = MAX_DEPTH
            plt.imsave(img_path, 1/gt_disp, cmap='magma')
            """
    print("\n-> Done!")


#""""
if __name__ == "__main__":
    cfg_path = "D:\\train_monodepth2_disentangle_full_hp3\cfg_kitti_monodepth2_disentangle.py"  # path to cfg file
    model_path ="D:\\train_monodepth2_disentangle_full_hp3\\epoch_13.pth"  # path to model weight
    gt_path = 'F:\\kitti\\Raw_data\\gt_depths.npz'#path to kitti gt depth
    output_path = './train_monodepth2_disentangle_full_hp3_test'   # dir for saving depth maps
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    evaluate(cfg_path,model_path,gt_path,output_path)
#"""

""""
if __name__ == "__main__":
    cfg_path = "./config/cfg_kitti_fm.py"  # path to model weight
    model_path = "D:\\VISUALIZATION FEATDEPTH\\fm_depth.pth" # path to cfg file
    gt_path = 'F:\\kitti\\Raw_data\\gt_depths.npz'#path to kitti gt depth
    output_path = './fm_depth'   # dir for saving depth maps
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    evaluate(cfg_path,model_path,gt_path,output_path)
"""