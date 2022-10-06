from __future__ import absolute_import, division, print_function
import torch.nn as nn
from .layers import ConvBlock, Conv3x3, upsample
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=3, num_ch_dec=(16, 32, 64, 128, 256)):
        super(Decoder, self).__init__()
        self.num_ch_dec = num_ch_dec
        # upconv
        self.upconv5 = ConvBlock(num_ch_enc[4], num_ch_dec[4])
        self.upconv4 = ConvBlock(num_ch_dec[4], num_ch_dec[3])
        self.upconv3 = ConvBlock(num_ch_dec[3], num_ch_dec[2])
        self.upconv2 = ConvBlock(num_ch_dec[2], num_ch_dec[1])
        self.upconv1 = ConvBlock(num_ch_dec[1], num_ch_dec[0])

        # iconv
        self.iconv5 = ConvBlock(num_ch_dec[4], num_ch_dec[4])
        self.iconv4 = ConvBlock(num_ch_dec[3], num_ch_dec[3])
        self.iconv3 = ConvBlock(num_ch_dec[2], num_ch_dec[2])
        self.iconv2 = ConvBlock(num_ch_dec[1], num_ch_dec[1])
        self.iconv1 = ConvBlock(num_ch_dec[0], num_ch_dec[0])

        # disp
        self.disp4 = Conv3x3(num_ch_dec[3], num_output_channels)
        self.disp3 = Conv3x3(num_ch_dec[2], num_output_channels)
        self.disp2 = Conv3x3(num_ch_dec[1], num_output_channels)
        self.disp1 = Conv3x3(num_ch_dec[0], num_output_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        _, _, _, _, econv5 = input_features
        # (64,64,128,256,512)*4

        upconv5 = upsample(self.upconv5(econv5))
        iconv5 = self.iconv5(upconv5)

        upconv4 = upsample(self.upconv4(iconv5))
        iconv4 = self.iconv4(upconv4)

        upconv3 = upsample(self.upconv3(iconv4))
        iconv3 = self.iconv3(upconv3)

        upconv2 = upsample(self.upconv2(iconv3))
        iconv2 = self.iconv2(upconv2)

        upconv1 = upsample(self.upconv1(iconv2))
        iconv1 = self.iconv1(upconv1)

        self.outputs[("res_img", frame_id, 3)] = self.sigmoid(self.disp4(iconv4))
        self.outputs[("res_img", frame_id, 2)] = self.sigmoid(self.disp3(iconv3))
        self.outputs[("res_img", frame_id, 1)] = self.sigmoid(self.disp2(iconv2))
        self.outputs[("res_img", frame_id, 0)] = self.sigmoid(self.disp1(iconv1))
        return self.outputs


class ColorDecoder(Decoder):
    def __init__(self, num_ch_enc, num_output_channels=3, skip_connection_multiplier=1):
        super(ColorDecoder, self).__init__(num_ch_enc, num_output_channels, num_ch_dec=(16, 32, 64, 128, 256))
        self.skip_connection_multiplier = skip_connection_multiplier

        self.upconv5_skip = ConvBlock(num_ch_enc[3], self.num_ch_dec[3])
        self.upconv4_skip = ConvBlock(num_ch_enc[2], self.num_ch_dec[2])
        self.upconv3_skip = ConvBlock(num_ch_enc[1], self.num_ch_dec[1])
        self.upconv2_skip = ConvBlock(num_ch_enc[0], self.num_ch_dec[0])

    def forward(self, input_features, outputs=None, frame_id=0, skip_layers=(None, None, None, None)):
        econv1, econv2, econv3, econv4, econv5 = input_features
        disp4 = outputs[("disp", frame_id, 3)]
        disp3 = outputs[("disp", frame_id, 2)]
        disp2 = outputs[("disp", frame_id, 1)]
        disp1 = outputs[("disp", frame_id, 0)]
        # (64,64,128,256,512)*4

        upconv5 = upsample(self.upconv5(econv5))
        _, _, h, w = upconv5.size()
        disp4 = F.interpolate(disp4, [h, w], mode="bilinear", align_corners=False)
        iconv5 = self.iconv5(upconv5) + disp4 * self.skip_connection_multiplier

        upconv4 = upsample(self.upconv4(iconv5))
        if skip_layers[0] and skip_layers[0] is not None:
            upconv4 = upconv4 + upsample(self.upconv5_skip(econv4))
        _, _, h, w = upconv4.size()
        disp3 = F.interpolate(disp3, [h, w], mode="bilinear", align_corners=False)
        iconv4 = self.iconv4(upconv4) + disp3 * self.skip_connection_multiplier

        upconv3 = upsample(self.upconv3(iconv4))
        if skip_layers[1] and skip_layers[1] is not None:
            upconv3 = upconv3 + upsample(self.upconv4_skip(econv3))
        _, _, h, w = upconv3.size()
        disp2 = F.interpolate(disp2, [h, w], mode="bilinear", align_corners=False)
        iconv3 = self.iconv3(upconv3) + disp2 * self.skip_connection_multiplier

        upconv2 = upsample(self.upconv2(iconv3))
        if skip_layers[2] and skip_layers[2] is not None:
            upconv2 = upconv2 + upsample(self.upconv3_skip(econv2))
        _, _, h, w = upconv2.size()
        disp1 = F.interpolate(disp1, [h, w], mode="bilinear", align_corners=False)
        iconv2 = self.iconv2(upconv2) + disp1 * self.skip_connection_multiplier

        upconv1 = upsample(self.upconv1(iconv2))
        if skip_layers[3] and skip_layers[3] is not None:
            upconv1 = upconv1 + upsample(self.upconv2_skip(econv1))
        iconv1 = self.iconv1(upconv1)

        outputs[("auto_res_img", frame_id, 3)] = self.sigmoid(self.disp4(iconv4))
        outputs[("auto_res_img", frame_id, 2)] = self.sigmoid(self.disp3(iconv3))
        outputs[("auto_res_img", frame_id, 1)] = self.sigmoid(self.disp2(iconv2))
        outputs[("auto_res_img", frame_id, 0)] = self.sigmoid(self.disp1(iconv1))
        return outputs