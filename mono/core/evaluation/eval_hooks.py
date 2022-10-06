import os
import os.path as osp
import cv2
import torch.nn.functional as F

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.parallel import scatter, collate
from torch.utils.data import Dataset
from .pixel_error import *
from .segmentation_metrics import SegmentationRunningScore

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

SEG_CLASS_WEIGHTS = (
    2.8149201869965, 6.9850029945374, 3.7890393733978, 9.9428062438965,
    9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606,
    4.6323022842407, 9.5608062744141, 7.8698215484619, 9.5168733596802,
    10.373730659485, 6.6616044044495, 10.260489463806, 10.287888526917,
    10.289801597595, 10.405355453491, 10.138095855713, 0
)
weights = torch.tensor(SEG_CLASS_WEIGHTS, device='cpu')

def change_input_variable(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if 'kp' not in k:
                data[k] = torch.as_tensor(v).float().cuda()
    else:
        img_data = data[0]
        for ind, (img) in enumerate(img_data):
            img_data[ind] = torch.as_tensor(img).float().cuda()
        data[0] = img_data
    return data


def unsqueeze_input_variable(data):
    for k, v in data.items():
        data[k] = torch.unsqueeze(v, dim=0)
    return data


class NonDistSegmentationEvalHook(Hook):
    def __init__(self, dataset, cfg):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = cfg.get('interval', 1)
        self.out_path = cfg.get('work_dir', './')
        self.cfg = cfg
        self.scores = SegmentationRunningScore(n_classes=cfg.num_classes)

    def after_train_epoch(self, runner):
        print('segmentation evaluation..............................................')

        val_loss = AverageMeter()
        self.scores.reset()
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()

        for idx in range(self.dataset.__len__()):
            data = list(self.dataset[idx])
            data[0] = torch.as_tensor(data[0]).float().cuda()
            # data = change_input_variable(data)
            data[0] = torch.unsqueeze(data[0], dim=0) # unsqueeze_input_variable(data)
            with torch.no_grad():
                result = runner.model(data)
            gt_depth = data[1]
            result = F.interpolate(result, gt_depth[0, :, :].shape, mode='nearest')  # upscale predictions
            """"
            if runner.model.module.weights is not None:
                loss_val = F.cross_entropy(result.cpu(), gt_depth.long(), weight=weights)
            else:
                loss_val = F.cross_entropy(result.cpu(), gt_depth.long())
            val_loss.update(loss_val)
            #"""
            seg_pred = result.exp().cpu()  # exp preds and shift to CPU
            seg_pred = seg_pred.numpy()  # transform preds to np array
            seg_pred = seg_pred.argmax(1)  # get the highest score for classes per pixel
            seg_gt = gt_depth.cpu().numpy()  # transform gt to np array
            self.scores.update(seg_gt, seg_pred)

        metrics = self.scores.get_scores()

        miou = metrics['meaniou']
        acc = metrics['meanacc']

        print(f' miou: {miou:8.3f} | acc: {acc:8.3f}', flush=True)
        print('val_loss is ', val_loss.avg)


class NonDistEvalHook(Hook):
    def __init__(self, dataset, cfg):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = cfg.get('validate_interval', 1)
        self.out_path = cfg.get('work_dir', './')
        self.cfg = cfg

    def after_train_epoch(self, runner):
        print('evaluation..............................................')

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()
        scale = AverageMeter()
        ratios = []
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()

        for idx in range(self.dataset.__len__()):
            data = self.dataset[idx]
            data = change_input_variable(data)
            data = unsqueeze_input_variable(data)
            with torch.no_grad():
                result = runner.model(data)

            disp = result[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp)
            pred_disp = pred_disp.cpu()[0, 0].numpy()

            gt_depth = data['gt_depth'].cpu()[0].numpy()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = compute_errors(gt_depth, pred_depth)

            abs_rel.update(abs_rel_)
            sq_rel.update(sq_rel_)
            rmse.update(rmse_)
            rmse_log.update(rmse_log_)
            a1.update(a1_)
            a2.update(a2_)
            a3.update(a3_)
            scale.update(ratio)
            ratios.append(ratio)

        print('abs_rel is ', abs_rel.avg)
        print('sq_rel is ', sq_rel.avg)
        print('rmse is ', rmse.avg)
        print('rmse_log is ', rmse_log.avg)

        print('a1 is ', a1.avg)
        print('a2 is ', a2.avg)
        print('a3 is ', a3.avg)

        print('scale mean', scale.avg)
        print('scale std', np.std(ratios))

        runner.log_buffer.output['abs_rel'] = float(abs_rel.avg)
        runner.log_buffer.output['sq_rel'] = float(sq_rel.avg)
        runner.log_buffer.output['rmse'] = float(rmse.avg)
        runner.log_buffer.output['rmse_log'] = float(rmse_log.avg)
        runner.log_buffer.output['a1'] = float(a1.avg)
        runner.log_buffer.output['a2'] = float(a2.avg)
        runner.log_buffer.output['a3'] = float(a3.avg)
        runner.log_buffer.output['scale mean'] = float(scale.avg)
        runner.log_buffer.output['scale std'] = float(np.std(ratios))
        runner.log_buffer.ready = True


class DistEvalHook(Hook):
    def __init__(self, dataset, interval=1, cfg=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = interval
        self.cfg = cfg

    def after_train_epoch(self, runner):
        print('evaluation..............................................')

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))

        t = 0
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data = change_input_variable(data)

            data_gpu = scatter(collate([data], samples_per_gpu=1), [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                t1 = cv2.getTickCount()
                result = runner.model(data_gpu)
                t2 = cv2.getTickCount()
                t += cv2.getTickFrequency() / (t2-t1)

            disp = result[("disp", 0, 0)]

            pred_disp, _ = disp_to_depth(disp)
            pred_disp = pred_disp.cpu()[0, 0].numpy()

            gt_depth = data['gt_depth'].cpu().numpy()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            if self.cfg.data['stereo_scale']:
                pred_depth *= 36
            else:
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = compute_errors(gt_depth, pred_depth)
            # if runner.rank == 0:
            #     if idx % 5 == 0:
            #         img_path = os.path.join(self.cfg.work_dir, 'visual_{:0>4d}.png'.format(idx))
            #         vmax = np.percentile(pred_disp, 95)
            #         plt.imsave(img_path, pred_disp, cmap='magma', vmax=vmax)

            result = {}
            result['abs_rel'] = float(abs_rel_)
            result['sq_rel'] = float(sq_rel_)
            result['rmse'] = float(rmse_)
            result['rmse_log'] = float(rmse_log_)
            result['a1'] = float(a1_)
            result['a2'] = float(a2_)
            result['a3'] = float(a3_)
            result['scale'] = float(ratio)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()


        if runner.rank == 0:
            print('\n')
            print('FPS:', t/len(self.dataset))

            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self, runner, results):
        raise NotImplementedError


class DistEvalSegmentationHook(Hook):

    def __init__(self, dataset, interval=1, cfg=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = interval
        self.cfg = cfg
        self.scores = SegmentationRunningScore(n_classes=cfg.num_classes)

    def after_train_epoch(self, runner):
        print('segmentation evaluation..............................................')
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))

        t = 0
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            self.scores.reset()
            data = self.dataset[idx]
            data[0] = torch.as_tensor(data[0]).float().cuda()

            data_gpu = scatter(collate([data], samples_per_gpu=1), [torch.cuda.current_device()])[0]
            # compute output
            with torch.no_grad():
                t1 = cv2.getTickCount()
                result = runner.model(data_gpu)
                t2 = cv2.getTickCount()
                t += cv2.getTickFrequency() / (t2 - t1)

            gt_depth = data[1]
            result = F.interpolate(result, gt_depth[0, :, :].shape, mode='nearest')
            """"
            if runner.model.module.weights is not None:
                loss_val = F.cross_entropy(result.cpu(), gt_depth.long(), weight=weights)
            else:
                loss_val = F.cross_entropy(result.cpu(), gt_depth.long())
            val_loss.update(loss_val)
            """
            seg_pred = result.exp().cpu()  # exp preds and shift to CPU
            seg_pred = seg_pred.numpy()  # transform preds to np array
            seg_pred = seg_pred.argmax(1)  # get the highest score for classes per pixel
            seg_gt = gt_depth.cpu().numpy()  # transform gt to np array

            self.scores.update(seg_gt, seg_pred)
            result = self.scores.get_scores()
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            print('FPS:', t / len(self.dataset))

            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self, runner, results):
        if mmcv.is_str(results):
            assert results.endswith('.pkl')
            results = mmcv.load(results)
        elif not isinstance(results, list):
            raise TypeError('results must be a list of numpy arrays or a filename, not {}'.format(type(results)))


        val_loss = AverageMeter()
        mios = AverageMeter()
        accs = AverageMeter()

        print('results len is ', results.__len__())
        for result in results:
            mios.update(result['iou'])
            accs.update(result['acc'])

        runner.log_buffer.output['miou'] = mios.avg
        runner.log_buffer.output['acc'] = accs.avg
        runner.log_buffer.ready = True


class DistEvalMonoHook(DistEvalHook):
    def evaluate(self, runner, results):
        if mmcv.is_str(results):
            assert results.endswith('.pkl')
            results = mmcv.load(results)
        elif not isinstance(results, list):
            raise TypeError('results must be a list of numpy arrays or a filename, not {}'.format(type(results)))

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()
        scale = AverageMeter()

        print('results len is ', results.__len__())
        ratio = []
        for result in results:
            abs_rel.update(result['abs_rel'])
            sq_rel.update(result['sq_rel'])
            rmse.update(result['rmse'])
            rmse_log.update(result['rmse_log'])
            a1.update(result['a1'])
            a2.update(result['a2'])
            a3.update(result['a3'])
            scale.update(result['scale'])
            ratio.append(result['scale'])

        runner.log_buffer.output['abs_rel'] = abs_rel.avg
        runner.log_buffer.output['sq_rel'] = sq_rel.avg
        runner.log_buffer.output['rmse'] = rmse.avg
        runner.log_buffer.output['rmse_log'] = rmse_log.avg
        runner.log_buffer.output['a1'] = a1.avg
        runner.log_buffer.output['a2'] = a2.avg
        runner.log_buffer.output['a3'] = a3.avg
        runner.log_buffer.output['scale mean'] = scale.avg
        runner.log_buffer.output['scale std'] = np.std(ratio)
        runner.log_buffer.ready = True
