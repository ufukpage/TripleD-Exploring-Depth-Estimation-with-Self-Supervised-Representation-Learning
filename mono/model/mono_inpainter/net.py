from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt

from ..registry import MONO
from ..mono_autoencoder.net import autoencoder


@MONO.register_module
class inpainter(autoencoder):
    def __init__(self, options):
        super(inpainter, self).__init__(options)
        self.opt = options
        os.makedirs('./inpainter_test', exist_ok=True)
        self.save_fig = False

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        interval = 1000
        target = inputs[("color", 0, 0)]
        mask = inputs[("mask", 0, 0)]
        for i in range(5):
            f=features[i]
            smooth_loss = self.get_smooth_loss(f, target)
            loss_dict[('smooth_loss', i)] = smooth_loss/ (2 ** i)/5

        for scale in self.opt.scales:
            """
            initialization
            """
            pred = outputs[("disp", 0, scale)]

            _,_,h,w = pred.size()
            target = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)

            mask_resize = F.interpolate(mask, [h, w], mode="bilinear", align_corners=False)
            min_reconstruct_loss = self.compute_reprojection_loss(pred, target)
            min_reconstruct_loss = torch.sum(min_reconstruct_loss * (1 - mask_resize)) / torch.sum(1 - mask_resize)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/len(self.opt.scales)

            if self.count % interval == 0 and self.save_fig:
                img_path = os.path.join('./inpainter_test', 'auto_{:0>4d}_{}.png'.format(self.count // interval, scale))
                plt.imsave(img_path, pred[0].transpose(0,1).transpose(1,2).data.cpu().numpy())
                img_path = os.path.join('./inpainter_test', 'img_{:0>4d}_{}.png'.format(self.count // interval, scale))
                plt.imsave(img_path, target[0].transpose(0, 1).transpose(1, 2).data.cpu().numpy())

        self.count += 1
        return loss_dict
