from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50, resnet101


class Encoder(nn.Module):
    def __init__(self, num_layers, pretrained_path=None):
        super(Encoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))


        self.encoder = resnets[num_layers]()
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.encoder.load_state_dict(checkpoint, strict=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.features = []
        # for name, param in self.encoder.named_parameters():
        #     if 'bn' in name:
        #         param.requires_grad = False

    def forward(self, input_image, input_features=None):
        self.features = []
        if input_features is not None:
            econv1, econv2, econv3, econv4, econv5 = input_features
        else:
            econv1, econv2, econv3, econv4, econv5 = 0, 0, 0, 0, 0
        self.features.append(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(input_image))) + econv1)
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])) + econv2)
        self.features.append(self.encoder.layer2(self.features[-1]) + econv3)
        self.features.append(self.encoder.layer3(self.features[-1]) + econv4)
        self.features.append(self.encoder.layer4(self.features[-1]) + econv5)
        return self.features

