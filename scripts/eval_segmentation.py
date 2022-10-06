from __future__ import absolute_import, division, print_function
import sys
from mmcv import Config

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.model.registry import SEGMENTATION
import torch.nn.functional as F
from mono.core.evaluation import SegmentationRunningScore
from mono.datasets.get_dataset import get_test_segmentation_dataset
from mono.datasets.labels_file import labels_cityscape_seg

STEREO_SCALE_FACTOR = 36
MIN_DEPTH=1e-3
MAX_DEPTH=80


def evaluate(MODEL_PATH, CFG_PATH):

    cfg = Config.fromfile(CFG_PATH)

    dataset = get_test_segmentation_dataset(cfg.data, val=False)
    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)

    cfg.model['imgs_per_gpu'] = 1
    model = SEGMENTATION.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda()
    model.eval()

    num_classes = len(labels_cityscape_seg.gettrainid2label())
    scores = SegmentationRunningScore(num_classes)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for ind, inp in enumerate(inputs):
                inputs[ind] = inp.cuda()

            gt_depth = inputs[1]
            result = model(inputs)
            result = F.interpolate(result, gt_depth[0, 0, :, :].shape, mode='nearest')  # upscale predictions

            seg_pred = result.exp().cpu()  # exp preds and shift to CPU
            seg_pred = seg_pred.numpy()  # transform preds to np array
            seg_pred = seg_pred.argmax(1)  # get the highest score for classes per pixel
            seg_gt = gt_depth.cpu().numpy()  # transform gt to np array
            scores.update(seg_gt, seg_pred)

    metrics = scores.get_scores()

    miou = metrics['meaniou']
    acc = metrics['meanacc']

    print(f' miou: {miou:8.3f} | acc: {acc:8.3f}', flush=True)


if __name__ == "__main__":
    CFG_PATH = '../config/cfg_kitti_fm_joint_inpaint_segmentation.py'#path to cfg file
    MODEL_PATH = 'D:\\CMP784Projects\\FeatDepthW\\segmentation_cs_BaseSegmentationFeat\\latest.pth'#path to model weights
    evaluate(MODEL_PATH, CFG_PATH)