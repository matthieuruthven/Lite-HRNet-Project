# lite_hrnet_data_loading.py
# Code to understand Lite-HRNet data loading

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 5th September 2023

# Import required modules
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from mmcv import Config, DictAction
from mmpose.datasets import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids

    # Build training dataset
    datasets = [build_dataset(cfg.data.train)]

    ### Code from mmpose/apis/train.py (function train_model()) ###

    ## Training dataset

    # prepare data loaders
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            shuffle=False,
            dist=False,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in datasets]
    
    ## Validation dataset
    eval_cfg = cfg.get('evaluation', {})
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        drop_last=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                                **cfg.data.get('val_dataloader', {}))
    val_dataloader = build_dataloader(val_dataset, **dataloader_setting)

    ### ###

    ### New code ###

    # Iterate through data loader
    for idx, data in enumerate(data_loaders[0]):

        # Calculate mini-batch size
        mbs = data['img'].shape[0]

        # Preallocate lists for images, images with heatmaps overlaid and heatmaps
        img_list = []
        img_hmp_list = []
        hmp_list = []

        # For each image
        for idx in range(mbs):

            # Extract image
            tmp_img = data['img'][idx, ...]

            # Convert pixel values to between 0 and 255 and dtype to uint8
            tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min()) * 255
            tmp_img = tmp_img.to(torch.uint8)

            # Update img_list
            img_list.append(tmp_img)

            # NumPy version of image
            tmp_img_np = tmp_img.permute((1, 2, 0)).numpy()

            # Extract heatmaps
            tmp_hmp = data['target'][idx, ...]

            # Flatten heatmaps
            tmp_hmp = torch.sum(tmp_hmp, dim = 0)

            # Convert pixel values to between 0 and 255 and dtype to uint8
            tmp_hmp = (tmp_hmp - tmp_hmp.min()) / (tmp_hmp.max() - tmp_hmp.min()) * 255
            tmp_hmp = tmp_hmp.to(torch.uint8)

            # Convert heatmap to RGB
            tmp_hmp = cv2.applyColorMap(tmp_hmp.numpy(), cv2.COLORMAP_JET)
            tmp_hmp = cv2.cvtColor(tmp_hmp, cv2.COLOR_BGR2RGB)

            # Resize the heatmap if it has a different size to the image
            if tmp_hmp.shape != tmp_img_np.shape:
                tmp_hmp = cv2.resize(tmp_hmp, (tmp_img_np.shape[0], tmp_img_np.shape[0]))

            # Combine overlay heatmap on image
            tmp_img_hmp = cv2.addWeighted(tmp_hmp, 0.5, tmp_img_np, 0.5, 0)

            # Convert combined image to PyTorch tensor
            tmp_img_hmp = torch.from_numpy(tmp_img_hmp).permute((2, 0, 1))

            # Update img_hmp_list
            img_hmp_list.append(tmp_img_hmp)

            # Convert heatmap to PyTorch tensor
            tmp_hmp = torch.from_numpy(tmp_hmp).permute((2, 0, 1))

            # Update hmp_list
            hmp_list.append(tmp_hmp)

        # Combine img_list, img_hmp_list and hmp_list
        img_list += img_hmp_list
        img_list += hmp_list

        # Make a grid of images
        Grid = make_grid(img_list, nrow=mbs)

        # Show grid
        img = ToPILImage()(Grid)
        img.show()
        plt.show()


if __name__ == '__main__':
    main()