from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50, resnet101


class DepthEncoder(nn.Module):
    def __init__(self, num_layers, pretrained_path=None):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers]()
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.encoder.load_state_dict(checkpoint, strict=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # for name, param in self.encoder.named_parameters():
        #     if 'bn' in name:
        #         param.requires_grad = False

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features.append(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x))))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

    def convert_to_group(self, disentangle_layers, conv_groups=1):
        if disentangle_layers[0]:
            self.encoder.conv1.groups = conv_groups
        for ind, layer in enumerate(disentangle_layers[1:]):
            if layer:
                conv_layers = list(getattr(self.encoder, '{}{}'.format('layer', ind+1)).modules())
                for conv_layer in conv_layers:
                    if isinstance(conv_layer, nn.Conv2d):
                        conv_layer.groups = conv_groups
                        #conv_layer.in_channels //= conv_groups
                        #conv_layer.out_channels //= conv_groups
        return
