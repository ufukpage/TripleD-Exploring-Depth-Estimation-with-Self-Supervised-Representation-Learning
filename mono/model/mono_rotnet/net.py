from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.transforms import RandomCrop


from ..registry import MONO
from ..mono_autoencoder.net import autoencoder


def resize(img, size=224):
    return F.interpolate(img, (size, size), mode="bilinear", align_corners=False)


def rotation(inputs):
    batch = inputs.shape[0]
    target = torch.LongTensor(np.random.permutation([0, 1, 2, 3] * (int(batch / 4) + 1)))[:batch]
    target = target.cuda()
    image = torch.zeros_like(inputs)
    image.copy_(inputs)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(inputs[i, :, :, :], target[i], [1, 2])

    return image, target


@MONO.register_module
class rotnet(autoencoder):
    def __init__(self, options):
        super(rotnet, self).__init__(options)
        self.opt = options
        self.AverageHead = nn.AdaptiveAvgPool2d((1, 1))
        self.crop = RandomCrop(self.opt.pretext_resize)
        self.Decoder = nn.Linear(2048, options.pretext_label_size)
        self.rot_loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        color_rotated, rot_gt = rotation(self.crop(inputs[("color", 0, 0)]))
        features = self.Encoder(color_rotated)
        rot_predicts = self.Decoder(self.AverageHead(features[-1]).flatten(1))
        outputs = {'rot_predicts': rot_predicts, 'rot_gt': rot_gt}
        if self.training:
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        target = inputs[("color", 0, 0)]
        for i in range(5):
            f=features[i]
            smooth_loss = self.get_smooth_loss(f, target)
            loss_dict[('smooth_loss', i)] = smooth_loss/ (2 ** i)/5

        loss_dict['ssl_rot_loss'] = self.rot_loss(F.softmax(outputs['rot_predicts'], 0), outputs['rot_gt']) \
                                    * self.opt.pretext_weight
        return loss_dict
