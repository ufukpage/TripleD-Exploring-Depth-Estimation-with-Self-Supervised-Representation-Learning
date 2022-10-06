import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBlock, fSEModule, Conv3x3, Conv1x1, CRPBlock, upshuffle, upsample, Attention_Module
import numpy as np


class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc, use_shuffle=False):
        super(DepthDecoder, self).__init__()

        bottleneck = 256
        stage = 4
        self.do = nn.Dropout(p=0.5)
        self.use_shuffle = use_shuffle
        if use_shuffle:
            self.up1 = upshuffle(bottleneck, 2)
            self.up2 = upshuffle(bottleneck, 2)
            self.up3 = upshuffle(bottleneck, 2)
            self.up4 = upshuffle(bottleneck, 2)

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)

        self.iconv4 = Conv3x3(512, bottleneck)
        self.iconv3 = Conv3x3(bottleneck*2+1, bottleneck)
        self.iconv2 = Conv3x3(bottleneck*2+1, bottleneck)
        self.iconv1 = Conv3x3(bottleneck*2+1, bottleneck)

        self.crp4 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp3 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp2 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp1 = self._make_crp(bottleneck, bottleneck, stage)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)

        # disp
        self.disp4 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp3 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)
        x4 = F.leaky_relu(x4)
        x4 = self.crp4(x4)
        x4 = self.merge4(x4)
        x4 = F.leaky_relu(x4)
        if self.use_shuffle:
            x4 = self.up4(x4)
        else:
            x4 = upsample(x4)
        disp4 = self.disp4(x4)

        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, disp4), 1)
        x3 = self.iconv3(x3)
        x3 = F.leaky_relu(x3)
        x3 = self.crp3(x3)
        x3 = self.merge3(x3)
        x3 = F.leaky_relu(x3)
        if self.use_shuffle:
            x3 = self.up3(x3)
        else:
            x3 = upsample(x3)
        disp3 = self.disp3(x3)

        x2 = self.reduce2(l2)
        x2 = torch.cat((x2, x3 , disp3), 1)
        x2 = self.iconv2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.crp2(x2)
        x2 = self.merge2(x2)
        x2 = F.leaky_relu(x2)
        if self.use_shuffle:
            x2 = self.up2(x2)
        else:
            x2 = upsample(x2)
        disp2 = self.disp2(x2)

        x1 = self.reduce1(l1)
        x1 = torch.cat((x1, x2, disp2), 1)
        x1 = self.iconv1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.crp1(x1)
        x1 = self.merge1(x1)
        x1 = F.leaky_relu(x1)
        if self.use_shuffle:
            x1 = self.up2(x1)
        else:
            x1 = upsample(x1)
        disp1 = self.disp1(x1)

        self.outputs[("disp", frame_id, 3)] = disp4
        self.outputs[("disp", frame_id, 2)] = disp3
        self.outputs[("disp", frame_id, 1)] = disp2
        self.outputs[("disp", frame_id, 0)] = disp1

        return self.outputs


class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_shuffle=False, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.mobile_encoder = mobile_encoder
        self.use_shuffle = use_shuffle
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                    + self.num_ch_dec[row] * 2 * (col - 1),
                                                                    output_channel=self.num_ch_dec[row] * 2)
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                    + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    self.num_ch_enc[row] + self.num_ch_enc[row + 1] // 2 +
                    self.num_ch_dec[row] * 2 * (col - 1), self.num_ch_dec[row] * 2)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                                     self.num_ch_enc[row],
                                                                                     self.num_ch_dec[row + 1])
                else:
                    self.convs["X_" + index + "_downsample"] = Conv1x1(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row]
                                                                       + self.num_ch_dec[row + 1] * (col - 1),
                                                                       self.num_ch_dec[row + 1] * 2)
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2,
                                                                                     self.num_ch_dec[row + 1])

        if self.mobile_encoder:
            self.convs["dispConvScale0"] = Conv3x3(4, self.num_output_channels)
            self.convs["dispConvScale1"] = Conv3x3(8, self.num_output_channels)
            self.convs["dispConvScale2"] = Conv3x3(24, self.num_output_channels)
            self.convs["dispConvScale3"] = Conv3x3(40, self.num_output_channels)
        else:
            for i in range(4):
                self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features, frame_id=0):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](features["X_{}{}".format(row + 1, col - 1)]),
                    low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disp", frame_id, 0)] = self.sigmoid(self.convs["dispConvScale0"](x))
        outputs[("disp", frame_id, 1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disp", frame_id, 2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disp", frame_id, 3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))

        return outputs


class DIFFDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_shuffle=False):
        super(DIFFDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        self.use_shuffle = use_shuffle

        # decoder
        self.convs = nn.ModuleDict()

        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

            # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4], 2 * self.num_ch_dec[4], self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64, self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
            self.convs["72"] = Attention_Module(self.num_ch_enc[4], self.num_ch_enc[3] * 2, 256)
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128)
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64, 64)
            self.convs["9"] = Attention_Module(64, 64, 32)
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, frame_id=0):
        outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72, feature36)
        x18 = self.convs["18"](x36, feature18)
        x9 = self.convs["9"](x18, [feature64])
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))

        outputs[("disp", frame_id, 0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        outputs[("disp", frame_id, 1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        outputs[("disp", frame_id, 2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        outputs[("disp", frame_id, 3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        return outputs
