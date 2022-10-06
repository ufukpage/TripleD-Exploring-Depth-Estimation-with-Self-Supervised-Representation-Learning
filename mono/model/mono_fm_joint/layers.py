from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='nearest')
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='nearest')
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='nearest')
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='nearest')
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth #0.01
    max_disp = 1 / min_depth #10
    scaled_disp = min_disp + (max_disp - min_disp) * disp #(10-0.01)*disp+0.01
    depth = 1 / scaled_disp
    return scaled_disp, depth


class Backproject(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Backproject, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.cuda())
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones.cuda()], 1)
        return cam_points


class Project(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def init_subpixel(weight, upscale_factor_squared):
    co, ci, h, w = weight.shape
    c02 = co // upscale_factor_squared
    # initialize sub kernel
    k = torch.empty([c02, ci, h, w])
    nn.init.kaiming_normal_(k)
    # repeat upscale_factor_squared times
    return k.repeat_interleave(upscale_factor_squared, dim=0)


def upshuffle(in_planes, upscale_factor, custom_init=init_subpixel):
    model = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_planes, in_planes*upscale_factor**2, kernel_size=(3, 3), stride=(1, 1), padding=0),
                nn.PixelShuffle(upscale_factor),
                nn.ELU(inplace=True)
            )
    if custom_init is not None:
        init_weight = custom_init(model[1].weight, upscale_factor**2)
        model[1].weight.data.copy_(init_weight)
    return model


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.pad(out)
        out = self.nonlin(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=(1, 1), stride=(1, 1), bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), (3, 3))

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv5x5, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(2)
        else:
            self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 5)
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'pointwise'), Conv1x1(in_planes if (i == 0) else out_planes, out_planes, False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'pointwise'))(top)
            x = top + x
        return x


def compute_depth_errors(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeAndExcitationBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, (1, 1), padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, (1, 1), padding=0, bias=True),
        )

    def forward(self, x):
        return self.block(x)


# https://github.com/shawLyu/HR-Depth/blob/main/layers.py
class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))


class ChannelDescriptorLayer(nn.Module):
    def __init__(self):
        super(ChannelDescriptorLayer, self).__init__()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
        x_mean = spatial_sum / (x.size(2) * x.size(3))
        x_variance = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
        return x_variance.pow(0.5), x_mean


class AdaptivelyScaledCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AdaptivelyScaledCALayer, self).__init__()

        self.local_channel_descriptors = ChannelDescriptorLayer()

        self.saeb_mean = SqueezeAndExcitationBlock(channel, reduction=reduction)
        self.saeb_std = SqueezeAndExcitationBlock(channel, reduction=reduction)

        self.small_descriptor_bottleneck = nn.Sequential(nn.Conv2d(2*channel, 1*channel, (1, 1)),
                                                         nn.ReLU(inplace=True))

        self.saeb_final = SqueezeAndExcitationBlock(channel, reduction=reduction)

        self.gating_function = nn.Sigmoid()

    def forward(self, x):

        std_des, mean_des = self.local_channel_descriptors(x)

        # refined descriptors
        ref_std_des = self.saeb_std(std_des)
        ref_mean_des = self.saeb_mean(mean_des)

        # descriptor fusion
        fused_des = torch.cat((ref_std_des, ref_mean_des), 1)

        # descriptor bottleneck
        fused_des = self.small_descriptor_bottleneck(fused_des)

        # final mask
        fused_des = self.saeb_final(fused_des)
        mask = self.gating_function(fused_des)

        return x * mask


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


# Channel/Pixel Based Attention (CA) Layer
class CALayer(nn.Module):

    """
    if pix_att is True then it does not use avg pooling and, it works as pixel attention.
    if contrast_aware is True then it uses summation of average and sta of channel instead of average .
    """
    def __init__(self, channel, reduction=16, contrast_aware=False, pix_att=False):
        super(CALayer, self).__init__()

        self.pix_att = pix_att
        self.contrast_aware = contrast_aware

        if contrast_aware:
            self.local_descriptor = CALayer.rescaled_contrast_layer

        if not pix_att and not contrast_aware:
            # global average pooling: feature --> point
            self.local_descriptor = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_att = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, (1, 1), padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, (1, 1), padding=0, bias=True),
            nn.Sigmoid()
        )

    @staticmethod
    def rescaled_contrast_layer(F):
        assert (F.dim() == 4)
        F_mean = mean_channels(F)
        F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
        # return F_mean / F_variance.pow(0.5)
        # return - F_mean + F_variance
        return -F_mean / F_variance.pow(0.5) + F_variance.pow(0.5)

    def forward(self, x):
        if not self.pix_att or self.contrast_aware:
            y = self.local_descriptor(x)
            y = self.conv_att(y)
        else:
            y = self.conv_att(x)
        return x * y


class IdentityPartial(nn.Module):

    """
    part_ratio: embedding ratio part
    use_right: use right part of the starting from the ratio
    """
    def __init__(self, part_ratio=2, use_right=True):
        super(IdentityPartial, self).__init__()
        self.part_ratio = part_ratio
        self.use_right = use_right

    def forward(self, embedding):
        if self.use_right:
            return embedding[:, embedding.size(1) // self.part_ratio:, :, :]
        return embedding[:, :embedding.size(1) // self.part_ratio, :, :]


class SPM(nn.Module):
    """ Structure Perception Module https://github.com/kamiLight/CADepth-master/blob/main/networks/spm.py"""
    def __init__(self, in_dim):
        super(SPM, self).__init__()
        self.chanel_in = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = out + x

        return out


# https://github.com/brandleyzhou/DIFFNet/blob/main/hr_layers.py
class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        # self.sa = SpatialAttention()
        # self.cs = CS_Block(channel)
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        features = self.ca(features)
        # features = self.sa(features)
        # features = self.cs(features)

        return self.relu(self.conv_se(features))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature
