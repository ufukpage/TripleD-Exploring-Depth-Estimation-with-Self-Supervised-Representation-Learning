import numpy as np
import os
import cv2
import scipy
from mono.model.registry import MONO
from mmcv import Config
import torch
from mono.model.mono_baseline.layers import disp_to_depth
import os.path as osp
from scipy import io
from tqdm import tqdm

CFG_PATH = 'D:\\TripleDModels\\train_monodepth2_disentangle_hr_hp1\\cfg_kitti_monodepth2_disentangle_hr.py'
MODEL_PATH = 'D:\\TripleDModels\\train_monodepth2_disentangle_hr_hp1\\epoch_9.pth'

# CFG_PATH = '../config/cfg_kitti_fm_joint_inpaint.py'  # path to cfg file
# MODEL_PATH = 'D:\\CMP784Projects\\FeatDepth\\epoch_40.pth'  # path to model weights
main_path = "D:\\Downloads\\Make3D\\"


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.endswith('jpg'):
            yield file

depths_gt = []
images = []
ratio = 2
h_ratio = 1 / (1.33333 * ratio)
color_new_height = 1704 / 2
depth_new_height = 21
input_images = []
fname = osp.join(main_path, 'Test134')
for filename in files(fname):
    filename = filename[4:-4]
    mat = scipy.io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
    depths_gt.append(mat["Position3DGrid"][:, :, 3])

    image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
    image = image[ int((2272 - color_new_height)/2):int((2272 + color_new_height)/2),:,:]
    images.append(image[:, :, ::-1])

depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
depths_gt_cropped = map(lambda x: x[int((55 - 21)/2):int((55 + 21)/2),:], depths_gt)
depths_gt_cropped = list(depths_gt_cropped)


cfg = Config.fromfile(CFG_PATH)

model = MONO.module_dict[cfg.model['name']](cfg.model)
checkpoint = torch.load(MODEL_PATH)

model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith('Depth')}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
# model.load_state_dict(checkpoint['state_dict'], strict=False)
model.cuda()
model.eval()

errors = []
with torch.no_grad():
    for i, input_color in tqdm(enumerate( images)):
        inputs = {}
        input_color = cv2.resize(input_color / 255.0, (640, 192), interpolation=cv2.INTER_AREA)
        input_color = torch.tensor(input_color, dtype=torch.float).cuda().permute(2, 0, 1)[None, :, :, :]
        inputs["color_aug", 0, 0] = input_color
        outputs = model(inputs)
        disp = outputs[("disp", 0, 0)]

        pred_disp, _ = disp_to_depth(disp, 0.1, 100)
        pred_disp = pred_disp.squeeze().cpu().numpy()

        depth_gt = depths_gt_cropped[i]

        depth_pred = 1 / pred_disp
        depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(depth_gt > 0, depth_gt < 70)
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= np.median(depth_gt) / np.median(depth_pred)
        depth_pred[depth_pred > 70] = 70
        errors.append(compute_errors(depth_gt, depth_pred))
    mean_errors = np.mean(errors, 0)

print(("{:>8} | " * 4).format("abs_rel", "sq_rel", "rmse", "rmse_log"))
print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))