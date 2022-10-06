from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..mono_fm_joint.layers import Backproject, Project
from ..registry import MONO
from ..mono_fm_joint.net import mono_fm_joint
from ..mono_fm_joint.resnet import BasicBlock
from ..mono_fm_joint.depth_decoder import DepthDecoder, HRDepthDecoder
from ..mono_fm_joint.decoder import ColorDecoder
from ..mono_fm_joint.encoder import Encoder
import torchvision as tv
from .color_conversions import rgb2lab
import argparse
from ..mono_fm_joint.layers import Conv1x1, CALayer, AdaptivelyScaledCALayer, IdentityPartial
from torch.nn import BatchNorm2d as bn


@MONO.register_module
class mono_fm_joint_inpaint(mono_fm_joint):
    def __init__(self, options):
        super(mono_fm_joint_inpaint, self).__init__(options)
        self.use_perceptual = True
        if self.opt.get('freeze_extractor', False):
            for param in self.Encoder.parameters():
                param.requires_grad = False
        if self.opt.perception_weight == 0.:
            del self.Encoder
            del self.Decoder
            self.use_perceptual = False
        if self.opt.get('img_reconstruct_weight', 1) == 0:
            del self.Decoder

    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs["color_aug", 0, 0]))
        if self.training:
            outputs.update(self.predict_poses(inputs))
            features = None
            if self.use_perceptual:
                features = self.Encoder(inputs[("color", 0, 0)] * inputs[("mask", 0, 0)])
                if self.opt.get('img_reconstruct_weight', 1) != 0:
                    outputs.update(self.Decoder(features, 0))
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        target = inputs[("color", 0, 0)]
        mask = inputs[("mask", 0, 0)]

        if features is not None:
            for i in range(5):
                f = features[i]
                regularization_loss = self.get_feature_regularization_loss(f, target)
                loss_dict[('feature_regularization_loss', i)] = regularization_loss / (2 ** i) / 5

            outputs = self.generate_features_pred(inputs, outputs)  # this outputs outputs[("feature", -1/1, 0)]
            """
            minimum perceptional loss
            """
            perceptional_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                src_f = outputs[("feature", frame_id, 0)]
                tgt_f = features[0]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)

            min_perceptional_loss, outputs["min_index"] = torch.min(perceptional_loss, dim=1)
            loss_dict['min_perceptional_loss'] = self.opt.perception_weight * min_perceptional_loss.mean()

        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]

            reprojection_losses = []

            if features is not None and self.opt.get('img_reconstruct_weight', 1) != 0:
                """
                autoencoder / in - painting
                """
                res_img = outputs[("res_img", 0, scale)]
                _, _, h, w = res_img.size()
                target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
                mask_resize = F.interpolate(mask, [h, w], mode="bilinear", align_corners=False)
                img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)
                img_reconstruct_loss = torch.sum(img_reconstruct_loss * (1 - mask_resize)) / torch.sum(1 - mask_resize)
                loss_dict[('img_reconstruct_loss', scale)] = img_reconstruct_loss / len(self.opt.scales) * \
                                                             self.opt.get('img_reconstruct_weight', 1)

            """
            reconstruction/reprojection
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
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

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
            loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss / (2 ** scale) / len(
                self.opt.scales)

        return loss_dict


@MONO.register_module
class mono_fm_joint_inpaint_distill_gs(mono_fm_joint_inpaint):
    def __init__(self, options):
        super(mono_fm_joint_inpaint_distill_gs, self).__init__(options)
        if self.opt.get('use_normal', False):
            self.DepthToGray = nn.Sequential(BasicBlock(2, 32, use_residual=False),
                                             nn.Conv2d(32, 1, kernel_size=(1, 1)))
        else:
            self.DepthToGray = nn.Sequential(BasicBlock(1, 32), nn.Conv2d(32, 1, kernel_size=(1, 1)))

        self.to_gray = tv.transforms.Grayscale(num_output_channels=1) if not self.opt.get('use_lab', False) else \
            mono_fm_joint_inpaint_distill_gs.rgb_to_l

    def calculate_surface_normal(self, disp):
        _, depth = self.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        dx, dy = torch.gradient(depth.repeat(1, 2, 1, 1), dim=[3, 2])
        dx, dy = dx[:, 0, :, :].unsqueeze(1), dy[:, 0, :, :].unsqueeze(1)
        normal = torch.cat((-dx, -dy, torch.ones_like(depth)), 1)
        n = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        normal = normal / n
        return (normal + 1) / 2

    def compute_distill_gs_loss(self, inputs, outputs):
        loss_dict = {}
        if self.opt.d2g_weight > 0.:
            disp = outputs[("disp", 0, 0)]
            disp = F.interpolate(disp, [int(self.opt.height), int(self.opt.width)], mode="bilinear",
                                 align_corners=False)
            if self.opt.get('use_normal', False):
                normal = self.calculate_surface_normal(disp)
                disp = normal[:, :2, :, :]

            target = inputs[("color", 0, 0)]
            gt_gray = self.to_gray(target)
            mask = inputs.get(("mask", 0, 0), None)
            if not self.opt.get('use_mask', False) or mask is None:
                pred_gray = self.DepthToGray(disp)
                depth_to_gray_loss = self.compute_perceptional_loss(gt_gray, pred_gray)
            else:
                if self.opt.get('use_normal', False):
                    mask = mask[:, :2, :, :]
                else:
                    mask = mask[:, 0, :, :].unsqueeze(1)
                pred_gray = self.DepthToGray(disp * mask)
                depth_to_gray_loss = self.compute_perceptional_loss(gt_gray, pred_gray)
                depth_to_gray_loss = torch.sum(depth_to_gray_loss * (1 - mask)) / torch.sum(1 - mask)
            loss_dict['depth_to_gray_loss'] = depth_to_gray_loss * self.opt.d2g_weight
        return loss_dict

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_distill_gs, self).compute_losses(inputs, outputs, features)
        loss_dict.update(self.compute_distill_gs_loss(inputs, outputs))
        return loss_dict

    @staticmethod
    def rgb_to_l(rgb):
        # credits to https://github.com/richzhang/colorization-pytorch/blob/master/util/util.py
        mask = (rgb > .04045).type(torch.FloatTensor)
        if rgb.is_cuda:
            mask = mask.cuda()

        rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)
        y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]

        mask = (y > .008856).type(torch.FloatTensor)
        if y.is_cuda:
            mask = mask.cuda()
        yint = y ** (1 / 3.) * mask + (7.787 * y + 16. / 116.) * (1 - mask)
        L = 116. * yint - 16.
        return L.unsqueeze(1) / 100


@MONO.register_module
class mono_fm_joint_inpaint_distill_colorize(mono_fm_joint_inpaint_distill_gs):
    def __init__(self, options):
        super(mono_fm_joint_inpaint_distill_colorize, self).__init__(options)
        if self.opt.get('use_normal', False):
            self.ColorizeNet = nn.Sequential(BasicBlock(4, 32, use_residual=False),
                                             # BasicBlock(32, 32),
                                             nn.Conv2d(32, 2, kernel_size=(1, 1)))
        else:
            self.ColorizeNet = nn.Sequential(BasicBlock(2, 32, use_residual=False),
                                             # BasicBlock(32, 32),
                                             nn.Conv2d(32, 2, kernel_size=(1, 1)))

        del self.DepthToGray

        self.to_lab = rgb2lab

    def compute_distill_colorize_loss(self, inputs, outputs):
        loss_dict = {}
        if self.opt.colorize_weight > 0.:
            disp = outputs[("disp", 0, 0)]
            disp = F.interpolate(disp, [int(self.opt.height), int(self.opt.width)], mode="bilinear",
                                 align_corners=False)
            if self.opt.get('use_normal', False):
                normal = self.calculate_surface_normal(disp)
                disp = torch.cat((disp, normal[:, :2, :, :]), 1)

            target = inputs[("color", 0, 0)]
            lab_color = self.to_lab(target, argparse.Namespace(l_cent=50., l_norm=50., ab_norm=110.))
            gt_ab = lab_color[:, 1:, :, :]
            mask = inputs.get(("mask", 0, 0), None)
            disp = torch.cat((disp, lab_color[:, 0, :, :].unsqueeze(1)), 1)
            if not self.opt.get('use_mask', False) or mask is None:
                pred_gray = self.ColorizeNet(disp)
                colorize_loss = self.compute_perceptional_loss(gt_ab, pred_gray)
            else:
                mask = mask[:, 0, :, :].unsqueeze(1)
                if self.opt.get('use_normal', False):
                    mask = mask.expand(mask.size(0), 4, mask.size(2), mask.size(3))

                pred_gray = self.ColorizeNet(disp * mask)
                colorize_loss = self.compute_perceptional_loss(gt_ab, pred_gray)
                colorize_loss = torch.sum(colorize_loss * (1 - mask)) / torch.sum(1 - mask)
            loss_dict['colorize_loss'] = colorize_loss * self.opt.colorize_weight
        return loss_dict

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_distill_gs, self).compute_losses(inputs, outputs, features)
        loss_dict.update(self.compute_distill_colorize_loss(inputs, outputs))
        return loss_dict


@MONO.register_module
class mono_fm_joint_inpaint_disentangle_distill_sep_colorize(mono_fm_joint_inpaint):

    def __init__(self, options):
        super(mono_fm_joint_inpaint_disentangle_distill_sep_colorize, self).__init__(options)
        for ind, layer in enumerate(self.opt.disentangle_layers):
            if layer:
                self.DepthEncoder.num_ch_enc[ind] = self.DepthEncoder.num_ch_enc[ind] // 2
        if not self.opt.get('use_hr_depth', False):
            self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc, self.opt.get('depth_use_shuffle', False))
        else:
            self.DepthDecoder = HRDepthDecoder(self.DepthEncoder.num_ch_enc,
                                               use_shuffle=self.opt.get('depth_use_shuffle', False))

        self.ColorizeEncoder = Encoder(self.opt.get('colorize_num_layers', 50), self.opt.colorize_pretrained_path)
        self.ColorizeDecoder = ColorDecoder(self.ColorizeEncoder.num_ch_enc, num_output_channels=2,
                                            skip_connection_multiplier=options.get('skip_connection_multiplier', 1))
        self.to_lab = rgb2lab

    def forward(self, inputs):
        scene_embedding = self.DepthEncoder(inputs["color_aug", 0, 0])
        depth_embeddings = []
        for ind, layer in enumerate(self.opt.disentangle_layers):
            embedding = scene_embedding[ind]
            if layer:
                depth_embeddings.append(embedding[:, :embedding.size(1) // 2, :, :])
            else:
                depth_embeddings.append(embedding)
        outputs = self.DepthDecoder(depth_embeddings)
        if self.training:
            outputs.update(self.predict_poses(inputs))
            target = inputs[("color", 0, 0)]
            lab_color = self.to_lab(target, argparse.Namespace(l_cent=50., l_norm=50., ab_norm=110.))
            gt_ab = lab_color[:, 1:, :, :]
            input_gs = lab_color[:, 0, :, :].unsqueeze(1)
            input_gs = input_gs.expand(input_gs.size(0), 3, input_gs.size(2), input_gs.size(3))
            if self.opt.get("cond_encoder", False):
                grey_scale_embeddings = self.ColorizeEncoder(input_gs, depth_embeddings)
            else:
                grey_scale_embeddings = self.ColorizeEncoder(input_gs, None)

            outputs = self.ColorizeDecoder(grey_scale_embeddings, outputs)
            inputs['gt_ab'] = gt_ab
            features = None
            if self.use_perceptual:
                features = self.Encoder(inputs[("color", 0, 0)])
                if self.opt.get('img_reconstruct_weight', 1) != 0:
                    outputs.update(self.Decoder(features, 0))
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def compute_colorization_loss(self, inputs, outputs):
        loss_dict = {}
        if self.opt.colorize_weight > 0.:
            target = inputs['gt_ab']
            auto_res_img = outputs[("auto_res_img", 0, 0)]
            distill_colorize_loss = self.compute_perceptional_loss(target, auto_res_img)
            if self.opt.get('use_distill_mask', False):
                mask = inputs.get(("mask", 0, 0), None)
                mask = mask[:, 0, :, :].unsqueeze(1)
                distill_colorize_loss = torch.sum(distill_colorize_loss * (1 - mask)) / torch.sum(1 - mask)
            loss_dict['distill_colorize_loss'] = distill_colorize_loss * self.opt.colorize_weight
        return loss_dict

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_disentangle_distill_sep_colorize, self).\
            compute_losses(inputs, outputs, features)
        loss_dict.update(self.compute_colorization_loss(inputs, outputs))
        return loss_dict


@MONO.register_module
class mono_fm_joint_inpaint_disentangle_distill_sep_inpaint(mono_fm_joint_inpaint):

    def __init__(self, options):
        super(mono_fm_joint_inpaint_disentangle_distill_sep_inpaint, self).__init__(options)
        for ind, layer in enumerate(self.opt.disentangle_layers):
            if layer:
                self.DepthEncoder.num_ch_enc[ind] = self.DepthEncoder.num_ch_enc[ind] // 2
        if not self.opt.get('use_hr_depth', False):
            self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc, self.opt.get('depth_use_shuffle', False))
        else:
            self.DepthDecoder = HRDepthDecoder(self.DepthEncoder.num_ch_enc,
                                               use_shuffle=self.opt.get('depth_use_shuffle', False))

        self.InpaintEncoder = Encoder(self.opt.get('inpaint_num_layers', 50), self.opt.inpaint_pretrained_path)
        self.InpaintDecoder = ColorDecoder(self.InpaintEncoder.num_ch_enc, num_output_channels=3,
                                           skip_connection_multiplier=options.get('skip_connection_multiplier', 1))

    def forward(self, inputs):
        scene_embedding = self.DepthEncoder(inputs["color_aug", 0, 0])
        depth_embeddings = []
        for ind, layer in enumerate(self.opt.disentangle_layers):
            embedding = scene_embedding[ind]
            if layer:
                depth_embeddings.append(embedding[:, :embedding.size(1) // 2, :, :])
            else:
                depth_embeddings.append(embedding)
        outputs = self.DepthDecoder(depth_embeddings)
        if self.training:
            outputs.update(self.predict_poses(inputs))
            mask = inputs.get(("mask", 0, 0), None)
            cond_embedings = None
            if self.opt.get("cond_encoder", False):
                cond_embedings = depth_embeddings
            inpaint_embeddings = self.InpaintEncoder(inputs[("color", 0, 0)] * mask, cond_embedings) \
                if mask is not None else self.InpaintEncoder(inputs[("color", 0, 0), cond_embedings])
            outputs = self.InpaintDecoder(inpaint_embeddings, outputs)
            features = None
            if self.use_perceptual:
                features = self.Encoder(inputs[("color", 0, 0)])
                if self.opt.get('img_reconstruct_weight', 1) != 0:
                    outputs.update(self.Decoder(features, 0))
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def compute_inpaint_loss(self, inputs, outputs):
        loss_dict = {}
        if self.opt.inpaint_weight > 0.:
            target = inputs[("color", 0, 0)]
            auto_res_img = outputs[("auto_res_img", 0, 0)]
            distill_inpaint_loss = self.compute_perceptional_loss(target, auto_res_img)
            if self.opt.get('use_distill_mask', True):
                mask = inputs.get(("mask", 0, 0), None)
                mask = mask[:, 0, :, :].unsqueeze(1)
                distill_inpaint_loss = torch.sum(distill_inpaint_loss * (1 - mask)) / torch.sum(1 - mask)
            loss_dict['distill_inpaint_loss'] = distill_inpaint_loss * self.opt.inpaint_weight
        return loss_dict

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_disentangle_distill_sep_inpaint, self).\
            compute_losses(inputs, outputs, features)
        loss_dict.update(self.compute_inpaint_loss(inputs, outputs))
        return loss_dict


@MONO.register_module
class mono_fm_joint_inpaint_disentangle(mono_fm_joint_inpaint):
    def __init__(self, options):
        super(mono_fm_joint_inpaint_disentangle, self).__init__(options)
        self.depth_skip_type = self.opt.get('depth_skip_type', 'use_half')
        self.depth_disentangle_type = self.opt.get('depth_disentangle_type', 'use_half')
        self.color_skip_type = self.opt.get('color_skip_type', 'use_half')
        self.use_pfp = self.opt.get('use_pfp', False)

        num_ch_enc = []

        for ind, layer in enumerate(self.opt.disentangle_layers):
            if layer:
                layers = []
                if self.depth_skip_type == 'ca':
                    layers.append(CALayer(self.DepthEncoder.num_ch_enc[ind]))
                elif self.depth_skip_type == 'pa':
                    layers.append(CALayer(self.DepthEncoder.num_ch_enc[ind], pix_att=True))
                elif self.depth_skip_type == 'asca':
                    layers.append(AdaptivelyScaledCALayer(self.DepthEncoder.num_ch_enc[ind]))

                if self.depth_disentangle_type == 'use_half':
                    layers.extend([IdentityPartial(part_ratio=2, use_right=False)])
                else:
                    layers.extend([Conv1x1(self.DepthEncoder.num_ch_enc[ind], self.DepthEncoder.num_ch_enc[ind] // 2),
                                   bn(self.DepthEncoder.num_ch_enc[ind] // 2),
                                   nn.ELU()])
                layer = nn.Sequential(*layers)
                num_ch_enc.append(self.DepthEncoder.num_ch_enc[ind] // 2)
            else:
                if self.depth_skip_type == 'ca':
                    layer = CALayer(self.DepthEncoder.num_ch_enc[ind])
                elif self.depth_skip_type == 'pa':
                    layer = CALayer(self.DepthEncoder.num_ch_enc[ind], pix_att=True)
                elif self.depth_skip_type == 'asca':
                    layer = AdaptivelyScaledCALayer(self.DepthEncoder.num_ch_enc[ind])
                elif self.depth_skip_type == '1x1' and ind == len(self.opt.disentangle_layers)-1:
                    layer = nn.Sequential(Conv1x1(self.DepthEncoder.num_ch_enc[ind], self.DepthEncoder.num_ch_enc[ind]),
                                          bn(self.DepthEncoder.num_ch_enc[ind]),
                                          nn.ELU())
                else:
                    layer = nn.Identity()
                num_ch_enc.append(self.DepthEncoder.num_ch_enc[ind])
            setattr(self, '{}_{}'.format('depth_skip_layer', ind), layer)

        if not self.opt.get('use_hr_depth', False):
            self.DepthDecoder = DepthDecoder(num_ch_enc, self.opt.get('depth_use_shuffle', False))
        else:
            self.DepthDecoder = HRDepthDecoder(num_ch_enc,
                                               use_shuffle=self.opt.get('depth_use_shuffle', False))

        num_ch_enc = []
        self.opt['color_skip_layers'] = self.opt.get('color_skip_layers', (False, False, False, False))
        if self.color_skip_type == '1x1':
            ind = 0
            for ind, layer in enumerate(self.opt.color_skip_layers):
                if layer:
                    layer = nn.Sequential(Conv1x1(self.DepthEncoder.num_ch_enc[ind],
                                                  self.DepthEncoder.num_ch_enc[ind] // 2),
                                          bn(self.DepthEncoder.num_ch_enc[ind] // 2),
                                          nn.ELU())
                    num_ch_enc.append(self.DepthEncoder.num_ch_enc[ind] // 2)
                else:
                    layer = nn.Identity()
                    num_ch_enc.append(self.DepthEncoder.num_ch_enc[ind])
                setattr(self, '{}_{}'.format('color_skip_layer', ind), layer)

            setattr(self, '{}_{}'.format('color_skip_layer', ind + 1), nn.Identity())
            num_ch_enc.append(self.DepthEncoder.num_ch_enc[-1])
        else:
            for ind, layer in enumerate(self.opt.disentangle_layers):
                if layer:
                    num_ch_enc.append(self.DepthEncoder.num_ch_enc[ind] // 2)
                else:
                    num_ch_enc.append(self.DepthEncoder.num_ch_enc[ind])

        self.ColorDecoder = ColorDecoder(num_ch_enc, num_output_channels=3,
                                         skip_connection_multiplier=options.get('skip_connection_multiplier', 1))

    def forward(self, inputs):
        scene_embedding = self.DepthEncoder(inputs["color_aug", 0, 0])
        depth_embeddings, color_embeddings = [], []

        for ind, layer in enumerate(self.opt.disentangle_layers):
            depth_embedding = getattr(self, '{}_{}'.format('depth_skip_layer', ind))(scene_embedding[ind])
            depth_embeddings.append(depth_embedding)

        if self.color_skip_type == '1x1':
            ind = 0
            for ind, layer in enumerate(self.opt.color_skip_layers):
                color_embedding = getattr(self,  '{}_{}'.format('color_skip_layer', ind))(scene_embedding[ind])
                color_embeddings.append(color_embedding)
            color_embeddings.append(getattr(self,  '{}_{}'.format('color_skip_layer', ind + 1))(scene_embedding[-1]))
        else:
            for ind, layer in enumerate(self.opt.disentangle_layers):
                embedding = scene_embedding[ind]
                if layer:
                    color_embeddings.append(embedding[:, embedding.size(1) // 2:, :, :])
                else:
                    color_embeddings.append(embedding)

        outputs = self.DepthDecoder(depth_embeddings)
        if self.training:
            outputs = self.ColorDecoder(color_embeddings, outputs, skip_layers=self.opt.color_skip_layers)
            if self.opt.get('use_pfp', False):
                pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0], [192, 640],
                                                 mode="bilinear", align_corners=False)
                              for f_i in self.opt.frame_ids[1:]}
                pose_feats[0] = F.interpolate(outputs[("auto_res_img", 0, 0)], [192, 640],
                                                 mode="bilinear", align_corners=False)
                outputs.update(self.predict_poses(inputs, pose_feats))
            else:
                outputs.update(self.predict_poses(inputs))
            features = None
            if self.use_perceptual:
                features = self.Encoder(inputs[("color", 0, 0)])
                if self.opt.get('img_reconstruct_weight', 1) != 0:
                    outputs.update(self.Decoder(features, 0))
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def compute_auto_res_loss(self, inputs, outputs):
        loss_dict = {}
        if self.opt.auto_res_weight > 0.:
            target = inputs[("color", 0, 0)]
            auto_res_img = outputs[("auto_res_img", 0, 0)]
            auto_res_loss = self.compute_perceptional_loss(target, auto_res_img)
            loss_dict['auto_res_loss'] = auto_res_loss * self.opt.auto_res_weight
        return loss_dict

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_disentangle, self).compute_losses(inputs, outputs, features)
        loss_dict.update(self.compute_auto_res_loss(inputs, outputs))
        return loss_dict


@MONO.register_module
class mono_fm_joint_inpaint_disentangle_distill_colorize(mono_fm_joint_inpaint_distill_colorize,
                                                         mono_fm_joint_inpaint_disentangle):
    def __init__(self, options):
        # mono_fm_joint_inpaint_disentangle.__init__(self, options)
        # mono_fm_joint_inpaint_distill_colorize.__init__(self, options)
        super(mono_fm_joint_inpaint_disentangle_distill_colorize, self).__init__(options)
        # https://stackoverflow.com/questions/26927571/multiple-inheritance-in-python3-with-different-signatures
        # super tells what class to skip in the MRO, that current class will be skipped
        # https://stackoverflow.com/questions/63672144/diamond-inheritance-in-python

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_disentangle, self).compute_losses(inputs, outputs, features)
        loss_dict.update(self.compute_distill_colorize_loss(inputs, outputs))
        loss_dict.update(self.compute_auto_res_loss(inputs, outputs))
        return loss_dict


@MONO.register_module
class mono_fm_joint_inpaint_map_pose(mono_fm_joint_inpaint):
    def __init__(self, options):
        super(mono_fm_joint_inpaint_map_pose, self).__init__(options)
        self.pose_cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_map_cls = nn.Linear(self.PoseEncoder.num_ch_enc[-1], options.map_output)
        self.init_std = 0.001
        # self.init_weights()

    def predict_map_pose(self, inputs):
        inp = inputs[-1]
        inp = self.pose_cls_pool(inp)
        inp = inp.view(inp.size(0), -1)
        return self.pose_map_cls(inp)

    def init_weights(self):
        nn.init.normal_(self.pose_map_cls.weight, 0, self.init_std)
        nn.init.constant_(self.pose_map_cls.bias, 0)

    def predict_poses(self, inputs):
        outputs = {}
        # [192,640] for kitti
        pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0], [192, 640], mode="bilinear", align_corners=False)
                      for f_i in self.opt.frame_ids}
        map_masks = {f_i: F.interpolate(inputs["map_mask", f_i, 0], [192, 640], mode="bilinear", align_corners=False)
                     for f_i in self.opt.frame_ids[1:]}
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                map_mask = map_masks[f_i]

                map_params = inputs[('map_params', f_i, 0)]
                alpha1 = map_params[:, :, 1]
                alpha1 = alpha1.view(alpha1.size(0), 1, 1, -1)
                if map_params.size(2) == 2:
                    alpha2 = map_params[:, :, 1]
                else:
                    alpha2 = map_params[:, :, 2]
                alpha2 = alpha1.view(alpha2.size(0), 1, 1, -1)
                aug_sup = pose_feats[f_i] * map_mask * alpha1 + pose_feats[f_i] * (1 - map_mask)
                aug_sc = pose_feats[0] * map_mask * alpha2 + pose_feats[0] * (1 - map_mask)
                """"
                for index in range(aug_sup.shape[0]):
                    sup_img = aug_sup[index, : , :, :]
                    sc_img = aug_sc[index, :, :, :]

                    cv2.namedWindow('Mask', 0)
                    cv2.namedWindow('Sup Img', 0)
                    cv2.namedWindow('Sc Img', 0)
                    cv2.imshow('Mask', (map_mask[index, :, :, :].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
                    cv2.imshow('Sup Img', (sup_img.cpu().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
                    cv2.imshow('Sc Img', (sc_img.cpu().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
                    cv2.waitKey(int(10000 / 1))
                #"""
                if f_i < 0:
                    pose_inputs = [aug_sup, aug_sc]
                else:
                    pose_inputs = [aug_sc, aug_sup]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                axisangle, translation = self.PoseDecoder(pose_inputs)
                map_logits = self.predict_map_pose(pose_inputs)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0],
                                                                                     invert=(f_i < 0))
                outputs[("map_pose_logit", f_i, 0)] = map_logits

        return outputs

    def compute_losses(self, inputs, outputs, features):
        loss_dict = super(mono_fm_joint_inpaint_map_pose, self).compute_losses(inputs, outputs, features)
        for f_i in self.opt.frame_ids[1:]:
            map_pose_logit = outputs[("map_pose_logit", f_i, 0)]
            map_params = inputs[('map_params', f_i, 0)]
            labels = map_params[:, :, 0].long()
            loss_dict[('map_pose_loss', f_i)] = F.cross_entropy(map_pose_logit, labels.view(labels.size(0), )) * \
                                                self.opt.map_pose_weight

        return loss_dict


@MONO.register_module
class mono_fm_joint_equivariant_inpaint(mono_fm_joint_inpaint):
    def __init__(self, options):
        super(mono_fm_joint_equivariant_inpaint, self).__init__(options)

    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        mask = inputs[("mask", 0, 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs["inv_K"])
            pix_coords = self.project(cam_points, inputs["K"], T)  # [b,h,w,2]
            img = inputs[("color", frame_id, 0)]
            outputs[("color", frame_id, scale)] = F.grid_sample(img, pix_coords, padding_mode="border")

            cam_points = self.backproject(depth, inputs["K"])
            pix_coords = self.project(cam_points, inputs["inv_K"], T)
            outputs[("mask", frame_id, scale)] = F.grid_sample(mask, pix_coords, padding_mode="border", mode='nearest')

        return outputs

    def generate_features_pred(self, inputs, outputs):
        disp = outputs[("disp", 0, 0)]

        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            img = inputs[("color", frame_id, 0)]
            src_feats = self.Encoder(img)
            feats = []
            for src_f in src_feats:
                scaled_disp = F.interpolate(disp, [int(src_f.size(2)), int(src_f.size(3))], mode="bilinear",
                                            align_corners=False)
                _, depth = self.disp_to_depth(scaled_disp, self.opt.min_depth, self.opt.max_depth)

                backproject = Backproject(self.opt.imgs_per_gpu, int(src_f.size(2)), int(src_f.size(3)))
                project = Project(self.opt.imgs_per_gpu, int(src_f.size(2)), int(src_f.size(3)))
                K = inputs[("K")].clone()
                K[:, 0, :] //= disp.size(2) // src_f.size(2)
                K[:, 1, :] //= disp.size(3) // src_f.size(3)
                inv_K = torch.zeros_like(K)
                for i in range(inv_K.shape[0]):
                    inv_K[i, :, :] = torch.pinverse(K[i, :, :])

                cam_points = backproject(depth, inv_K)
                pix_coords = project(cam_points, K, T)  # [b,h,w,2]
                transformed_src_f = F.grid_sample(src_f, pix_coords, padding_mode="border")
                feats.append(transformed_src_f)
            outputs[("feature", frame_id)] = feats
        return outputs

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        target = inputs[("color", 0, 0)]
        mask = inputs[("mask", 0, 0)]

        for i in range(5):
            f = features[i]
            regularization_loss = self.get_feature_regularization_loss(f, target)
            loss_dict[('feature_regularization_loss', i)] = regularization_loss / (2 ** i) / 5

        outputs = self.generate_features_pred(inputs, outputs)  # this outputs outputs[("feature", -1/1, 0)]
        for frame_id in self.opt.frame_ids[1:]:
            src_feats = outputs[("feature", frame_id)]
            outputs.update(self.Decoder(src_feats, frame_id))

        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]

            reprojection_losses = []
            equivariant_losses = []
            """
            autoencoder / in - painting
            """
            res_img = outputs[("res_img", 0, scale)]
            _, _, h, w = res_img.size()
            target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            mask_resize = F.interpolate(mask, [h, w], mode="bilinear", align_corners=False)
            img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)
            img_reconstruct_loss = torch.sum(img_reconstruct_loss * (1 - mask_resize)) / torch.sum(1 - mask_resize)
            loss_dict[('img_reconstruct_loss', scale)] = img_reconstruct_loss / len(self.opt.scales)

            """
            reconstruction/reprojection
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
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
             minimum equivariant loss
             """
            for frame_id in self.opt.frame_ids[1:]:
                mask_transformed = outputs[("mask", frame_id, scale)]
                res_img = outputs[("res_img", frame_id, scale)]
                _, _, h, w = res_img.size()
                target_resize = F.interpolate(inputs[("color", frame_id, 0)], [h, w], mode="bilinear",
                                              align_corners=False)
                mask_resize = F.interpolate(mask_transformed, [h, w], mode="bilinear", align_corners=False)
                img_equivariant_loss = self.compute_reprojection_loss(res_img, target_resize)
                img_equivariant_loss = torch.sum(img_equivariant_loss * (1 - mask_resize)) / torch.sum(1 - mask_resize)
                equivariant_losses.append(img_equivariant_loss)

            equivariant_loss = torch.cat(equivariant_losses, 1)
            min_equivariant_loss, outputs["min_index"] = torch.min(equivariant_loss, dim=1)
            loss_dict['min_equivariant_loss', scale] = self.opt.equivariant_weight * min_equivariant_loss.mean() \
                                                       / len(self.opt.scales)

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
            loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss / (2 ** scale) / len(
                self.opt.scales)

        return loss_dict
