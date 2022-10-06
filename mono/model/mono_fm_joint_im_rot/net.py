from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn

from ..mono_fm_joint.layers import Backproject, Project

from ..registry import MONO
import numpy as np

from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF
from ..mono_fm_joint.net import mono_fm_joint


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
class mono_fm_joint_im_rot(mono_fm_joint):
    def __init__(self, options):
        super(mono_fm_joint_im_rot, self).__init__(options)
        self.AverageHead = nn.AdaptiveAvgPool2d((1, 1))
        self.ClassificationHead = nn.Linear(2048, options.pretext_label_size)
        self.rot_loss = nn.CrossEntropyLoss()
        self.crop = RandomCrop(self.opt.pretext_resize)

    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs["color_aug", 0, 0]))
        if self.training:
            outputs.update(self.predict_poses(inputs))
            color_rotated, rot_gt = rotation(self.crop(inputs[("color", 0, 0)]))
            features = self.Encoder(color_rotated)
            rot_pred = self.ClassificationHead(self.AverageHead(features[-1]).flatten(1))

            loss_dict = self.compute_losses(inputs, outputs, features, rot_pred, rot_gt)
            return outputs, loss_dict
        return outputs

    def compute_losses(self, inputs, outputs, features, rot_predicts=None, rot_gt=None):
        loss_dict = {}
        target = inputs[("color", 0, 0)]

        """
        rotation ssl loss
        """
        loss_dict['ssl_rot_loss'] = self.rot_loss(F.softmax(rot_predicts, 0), rot_gt) * self.opt.pretext_weight

        w, h = target.shape[-1], target.shape[-2]
        th, tw = self.opt.pretext_resize, self.opt.pretext_resize
        self.rand_i = torch.randint(0, h - th + 1, size=(1, )).item()
        self.rand_j = torch.randint(0, w - tw + 1, size=(1, )).item()

        for i in range(5):
            f=features[i]
            regularization_loss = self.get_feature_regularization_loss(f, target)
            loss_dict[('feature_regularization_loss', i)] = regularization_loss/(2 ** i)/5

        outputs = self.generate_features_pred(inputs, outputs)
        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]

            reprojection_losses = []
            perceptional_losses = []

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            if self.opt.automask:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 1e-5
                    reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, outputs[("min_index", scale)] = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/len(self.opt.scales)

            """
            minimum perceptional loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                src_f = outputs[("feature", frame_id, 0)]
                tgt_f = self.Encoder(TF.crop(inputs[("color", 0, 0)], self.rand_i, self.rand_j, th, tw))[0]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)

            min_perceptional_loss, outputs[("min_index", scale)] = torch.min(perceptional_loss, dim=1)
            loss_dict[('min_perceptional_loss', scale)] = self.opt.perception_weight * min_perceptional_loss.mean() / len(self.opt.scales)

            """
            disp mean normalization
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = self.get_smooth_loss(disp, target)
            loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss / (2 ** scale)/len(self.opt.scales)

        return loss_dict

    def generate_features_pred(self, inputs, outputs):
        disp = outputs[("disp", 0, 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        disp = TF.crop(disp, self.rand_i, self.rand_j, self.opt.pretext_resize, self.opt.pretext_resize)
        disp = F.interpolate(disp, [int(self.opt.pretext_resize/2), int(self.opt.pretext_resize/2)], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            backproject = Backproject(self.opt.imgs_per_gpu, int(self.opt.pretext_resize / 2), int(self.opt.pretext_resize / 2))
            project = Project(self.opt.imgs_per_gpu, int(self.opt.pretext_resize / 2), int(self.opt.pretext_resize / 2))

            K = inputs[("K")].clone()
            K[:, 0, :] /= 2
            K[:, 1, :] /= 2

            inv_K = torch.zeros_like(K)
            for i in range(inv_K.shape[0]):
                inv_K[i, :, :] = torch.pinverse(K[i, :, :])

            cam_points = backproject(depth, inv_K)
            pix_coords = project(cam_points, K, T)  # [b,h,w,2]

            img = inputs[("color", frame_id, 0)]
            src_f = self.Encoder(TF.crop(img, self.rand_i, self.rand_j, self.opt.pretext_resize, self.opt.pretext_resize))[0]
            outputs[("feature", frame_id, 0)] = F.grid_sample(src_f, pix_coords, padding_mode="border")
        return outputs
