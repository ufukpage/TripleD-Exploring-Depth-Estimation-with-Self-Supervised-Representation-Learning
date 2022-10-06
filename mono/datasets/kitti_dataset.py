from __future__ import absolute_import, division, print_function

import os
import scipy.misc
import numpy as np
import PIL.Image as pil
import datetime
import torch
import torch.utils.data as data
from .kitti_utils import generate_depth_map, read_calib_file, transform_from_rot_trans, pose_from_oxts_packet
from .mono_dataset import MonoDataset
import random
import cv2

class KittiSegmentation(data.Dataset):
    DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
    DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
    '''
    Dataset Class for KITTI Semantic Segmentation Benchmark dataset
    Dataset link - http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    There are 34 classes in the given labels. However, not all of them are useful for training
    (like railings on highways, road dividers, etc.).
    So, these useless classes (the pixel values of these classes) are stored in the `void_labels`.
    The useful classes are stored in the `valid_labels`.

    The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
    (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
    `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
    the loss function when comparing with the output.

    The `get_filenames` function retrieves the filenames of all images in the given `path` and
    saves the absolute path in a list.

    In the `get_item` function, images and masks are resized to the given `img_size`, masks are
    encoded using `encode_segmap`, and given `transform` (if any) are applied to the image only
    (mask does not usually require transforms, but they can be implemented in a similar way).
    '''
    IMAGE_PATH = os.path.join('training', 'image_2')
    MASK_PATH = os.path.join('training', 'semantic')

    def __init__(
        self,
        data_path,
        split,
        img_size=(1242, 376),
        void_labels=DEFAULT_VOID_LABELS,
        valid_labels=DEFAULT_VALID_LABELS,
        transform=None
    ):
        self.img_size = img_size
        self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.split = split
        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, 'training/image_2')
        self.mask_path = os.path.join(self.data_path, 'training/semantic')
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        # Split between train and valid set
        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        if self.split == 'train':
            idxs = [idx for idx in range(n_items) if idx not in idxs]
        elif self.split == 'test':
            idxs = [idx for idx in range(n_items)]

        # if self.split == 'train': idxs = [idx for idx in range(n_items)] # use them all
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]

    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        img = pil.open(self.img_list[idx])
        """"
        img = img.resize(self.img_size)
        img = np.array(img)
        # """
        mask = pil.open(self.mask_list[idx]).convert('L')
        """"
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)
        #"""
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

    def encode_segmap(self, mask):
        '''
        Sets void classes to zero so they won't be considered for training
        '''
        for voidc in self.void_labels:
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        mask[mask>18]=self.ignore_index
        return mask

    def get_filenames(self, path):
        '''
        Returns a list of absolute paths to images inside given `path`
        '''
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIInpaintDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIInpaintDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def preprocess_masks(self, inputs):
        image = inputs[("color", 0, 0)]

        mask = torch.ones(image.shape, dtype=torch.uint8)
        if self.cfg.erase_count == 1:  ### erase a patch in the center of image
            offset = (image.shape[1] - self.cfg.erase_shape[0]) / 2
            end = offset + self.cfg.erase_shape[0]
            mask[:, offset:end, offset:end] = 0
        else:
            for c_ in range(self.cfg.erase_count):
                row = torch.LongTensor(1).random_(0, image.shape[1] - self.cfg.erase_shape[0] - 1)[0]
                col = torch.LongTensor(1).random_(0, image.shape[2] - self.cfg.erase_shape[1] - 1)[0]

                mask[:, row:row + self.cfg.erase_shape[0], col:col + self.cfg.erase_shape[1]] = 0

        inputs[("mask", 0, 0)] = mask

    def preprocess(self, inputs, color_aug):
        super(KITTIInpaintDataset, self).preprocess(inputs, color_aug)
        self.preprocess_masks(inputs)


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = scipy.misc.imresize(depth_gt, self.full_res_shape[::-1], "nearest")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index, offset):
        oxts_root = os.path.join(self.data_path, folder, 'oxts')
        with open(os.path.join(oxts_root, 'timestamps.txt')) as f:
            timestamps = np.array([datetime.datetime.strptime(ts[:-3], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                                   for ts in f.read().splitlines()])

        speed0 = np.genfromtxt(os.path.join(oxts_root, 'data', '{:010d}.txt'.format(frame_index)))[[8, 9, 10]]
        # speed1 = np.genfromtxt(os.path.join(oxts_root, 'data', '{:010d}.txt'.format(frame_index+offset)))[[8, 9, 10]]

        timestamp0 = timestamps[frame_index]
        timestamp1 = timestamps[frame_index+offset]
        # displacement = 0.5 * (speed0 + speed1) * (timestamp1 - timestamp0)
        displacement = speed0 * (timestamp1 - timestamp0)

        imu2velo = read_calib_file(os.path.join(self.data_path, os.path.dirname(folder), 'calib_imu_to_velo.txt'))
        velo2cam = read_calib_file(os.path.join(self.data_path, os.path.dirname(folder), 'calib_velo_to_cam.txt'))
        cam2cam = read_calib_file(os.path.join(self.data_path, os.path.dirname(folder), 'calib_cam_to_cam.txt'))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        odo_pose = imu2cam[:3,:3] @ displacement + imu2cam[:3,3]

        return odo_pose


class KITTIMAPDataset(KITTIInpaintDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIMAPDataset, self).__init__(*args, **kwargs)
        self.map_cfg = kwargs['cfg'].map_cfg

    def get_map_params(self):

        alphas = self.map_cfg.get('alphas')
        if self.map_cfg.get('map_n', 1) == 1:
            max_pos = len(alphas)
            gt_map = random.randint(0, max_pos - 1)
            return [gt_map, alphas[gt_map]]
        else:
            max_pos = len(alphas) ** 2
            gt_map = random.randint(0, max_pos - 1)
            ind1 = gt_map // len(alphas)  # flag_1 = 7 // 4 = 1  gt of first 8 frames
            ind2 = gt_map % len(alphas)  # flag_2 = 7 %  4 = 3  gt of last 8 framesmix2cls_transforms.py
            return [gt_map, alphas[ind1], alphas[ind2]]

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.resize(inputs[(n, im, - 1)])

        tar_im = inputs[('color', 0, 0)]
        inputs[('color', 0, 0)] = self.to_tensor(tar_im)
        inputs[("color_aug", 0, 0)] = self.to_tensor(color_aug(tar_im))
        target_im = np.asarray(tar_im)
        target_gray = cv2.cvtColor(target_im, cv2.COLOR_BGR2GRAY)
        for f_i in self.frame_idxs[1:]:
            params = self.get_map_params()
            source_im = np.asarray(inputs[('color', f_i, 0)])
            source_gray = cv2.cvtColor(source_im, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(source_gray, target_gray)
            blurred_diff = cv2.GaussianBlur(diff, self.map_cfg.get('blur_kernel_size', (9, 9)), 0)

            if 'threshold' in self.map_cfg:
                threshold = self.map_cfg['threshold']
                mask = blurred_diff ; mask[blurred_diff <= threshold] = 0 ;  mask[blurred_diff > threshold] = 255
            else:
                ret, mask = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            mask = np.dstack([mask] * 3)
            inputs[('map_mask', f_i, 0)] = self.to_tensor(mask)
            inputs[('map_params', f_i, 0)] = torch.FloatTensor(params).view(1, -1)
            im = inputs[('color', f_i, 0)]
            inputs[('color', f_i, 0)] = self.to_tensor(im)
            inputs[("color_aug", f_i, 0)] = self.to_tensor(color_aug(im))
            """"
            alpha1 = params[1]
            if len(params) == 2:
                alpha2 = params[1]
            else:
                alpha2 = params[2]

            source_blended = ((source_im * (mask/255)) * alpha1).astype(np.uint8) + \
                              ((source_im * ((255-mask)/255).astype(np.uint8))) #* (1-alpha)).astype(np.uint8)

            target_blended = ((target_im * (mask/255)) * alpha2).astype(np.uint8) + \
                             ((target_im * ((255-mask)/255).astype(np.uint8))) #* (1-alpha)).astype(np.uint8)
            cv2.namedWindow('Mask', 0)
            cv2.namedWindow('Source Blended', 0)
            cv2.namedWindow('Target Blended', 0)
            cv2.namedWindow('Diff', 0)

            cv2.imshow('Diff', blurred_diff)
            cv2.imshow('Mask', mask)
            cv2.imshow('Source Blended', source_blended.astype(np.uint8))
            cv2.imshow('Target Blended', target_blended.astype(np.uint8))
            cv2.waitKey(int(100000/1))
            # """

        self.preprocess_masks(inputs)


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        side_map = {"l": 0, "r": 1}
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
