# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import warnings
from collections import OrderedDict
import pandas as pd

import numpy as np
from mmcv import Config, deprecated_api_warning
from scipy.io import loadmat, savemat

from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset
from mmpose.core.evaluation import keypoint_pck_accuracy


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


@DATASETS.register_module()
class TopDownSpeedPlusDataset(Kpt2dSviewRgbImgTopDownDataset):
    """SPEED+ Dataset for top-down pose estimation.

    "Next Generation Spacecraft Pose Estimation Dataset (SPEED+)"
    "https://zenodo.org/record/5588480"

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/speedplus.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            coco_style=False,
            test_mode=test_mode)

        self.db = self._get_db()
        self.image_set = set(x['image_file'] for x in self.db)
        self.num_images = len(self.image_set)

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        
        # Read file containing keypoint and bounding box information
        with open(self.ann_file) as anno_file:
            anno = json.load(anno_file)

        # Preallocate list
        gt_db = []

        # For each image
        for a in anno:
            
            # Extract image name
            image_name = a['filename']

            # Preallocate arrays for keypoint coordinates and
            # keypoint visibility labels
            joints_3d = np.zeros((self.ann_info['num_joints'], 3),
                                 dtype=np.float32)
            joints_3d_visible = np.zeros((self.ann_info['num_joints'], 3),
                                         dtype=np.float32)
            
            # Populate arrays
            if not self.test_mode:
                joints = np.array(a['keypoints']).T.reshape([-1, 3])
                assert joints.shape[0] == self.ann_info['num_joints'], \
                    f'joint num diff: {len(joints)}' + \
                    f' vs {self.ann_info["num_joints"]}'
                joints_3d[:, :2] = joints[:, :2]
                joints_3d_visible[:, :2] = np.expand_dims(joints[:, 2], axis=-1)
            
            # Path to file
            image_file = osp.join(self.img_prefix, image_name)
            
            # Update gt_db
            gt_db.append({
                'image_file': image_file,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox': np.array(a['bbox'], dtype=int)
            })

        return gt_db
    
    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCKh', **kwargs):
        """Evaluate PCKh for SPEED+ dataset. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - image_paths (list[str]): For example, ['/val2017/000000\
                    397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
            res_folder (str, optional): The folder to save the testing
                results. Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.

        Returns:
            dict: PCKh for each joint
        """

        # Check metrics
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCKh']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        # Create NumPy array of coordinates of keypoints estimated by the model
        est_coords = np.dstack([result['preds'][0, ...] for result in results])

        if res_folder:
            est_file = osp.join(res_folder, 'est_coords.mat')
            savemat(est_file, mdict={'est_coords': est_coords})

        SC_BIAS = 0.6
        threshold = 0.05 * self.ann_info['image_size'][0]

        # Read MAT file of GT keypoint locations
        gt_file = f'{self.ann_file[:-4]}mat'
        gt_dict = loadmat(gt_file)
        gt_coords = gt_dict['gt_coords']
        vis_kpts = gt_coords[:, -1, :]

        # Calculate difference between ground-truth coordinates
        # and estimated coordinates
        coord_error = est_coords[:, :-1, :] - gt_coords[:, :-1, :]
        coord_err = np.linalg.norm(coord_error, axis=1)

        # Count number of each keypoint
        kpt_count = np.sum(vis_kpts, axis=1)
        
        # Calculate percentage of correct keypoints
        less_than_threshold = (coord_err <= threshold) * vis_kpts
        PCK = 100. * np.sum(less_than_threshold, axis=1) / kpt_count
        PCK = np.ma.array(PCK, mask=False)
        kpt_count = np.ma.array(kpt_count, mask=False)
        kpt_ratio = kpt_count / np.sum(kpt_count).astype(np.float64)
        name_value = [('PCKh', np.sum(PCK * kpt_ratio))]
        name_value = OrderedDict(name_value)

        return name_value


@DATASETS.register_module()
class TopDownSpeedPlusDatasetPreProc(Kpt2dSviewRgbImgTopDownDataset):
    """Pre-processed SPEED+ Dataset for top-down pose estimation.

    "Next Generation Spacecraft Pose Estimation Dataset (SPEED+)"
    "https://zenodo.org/record/5588480"

    NB assumed pre-processing:
    1) images cropped and resized
    2) keypoint coordinates updated accordingly

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/speedplus.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            coco_style=False,
            test_mode=test_mode)

        self.db = self._get_db()
        self.image_set = set(x['image_file'] for x in self.db)
        self.num_images = len(self.image_set)

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        
        # Read file containing keypoint information
        with open(self.ann_file) as anno_file:
            anno = json.load(anno_file)

        # Preallocate list
        gt_db = []

        # For each image
        for a in anno:
            
            # Extract image name
            image_name = a['filename']

            # Preallocate arrays for keypoint coordinates and
            # keypoint visibility labels
            joints_3d = np.zeros((self.ann_info['num_joints'], 3),
                                 dtype=np.float32)
            joints_3d_visible = np.zeros((self.ann_info['num_joints'], 3),
                                         dtype=np.float32)
            
            # Populate arrays
            if not self.test_mode:
                joints = np.array(a['keypoints']).T.reshape([-1, 3])
                assert joints.shape[0] == self.ann_info['num_joints'], \
                    f'joint num diff: {len(joints)}' + \
                    f' vs {self.ann_info["num_joints"]}'
                joints_3d[:, :2] = joints[:, :2]
                joints_3d_visible[:, :2] = np.expand_dims(joints[:, 2], axis=-1)
            
            # Path to file
            image_file = osp.join(self.img_prefix, image_name)
            
            # Extract image dimensions
            image_size = self.ann_info['image_size']
            if image_size[0] == image_size[1]:
                image_size = image_size[0]
                
            # May need to add bounding box

            # Update gt_db
            gt_db.append({
                'image_file': image_file,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'center': np.array([image_size / 2, image_size / 2], dtype=np.float32),
                'scale': np.array([image_size / 200, image_size / 200], dtype=np.float32)
            })
            # NB division by 200 required because of scaling factor in transform_preds() 
            # in mmpose/core/post_processing/post_transforms.py
        
        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCKh', **kwargs):
        """Evaluate PCKh for SPEED+ dataset. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - image_paths (list[str]): For example, ['/val2017/000000\
                    397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
            res_folder (str, optional): The folder to save the testing
                results. Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.

        Returns:
            dict: PCKh for each joint
        """

        # Check metrics
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCKh', 'PCK']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        # Create NumPy array of coordinates of keypoints estimated by the model
        est_coords = np.dstack([result['preds'][0, ...] for result in results])

        threshold = 0.05 * self.ann_info['image_size'][0]

        # Read MAT file of GT keypoint locations
        gt_file = f'{self.ann_file[:-4]}mat'
        gt_dict = loadmat(gt_file)
        gt_coords = gt_dict['gt_coords']
        vis_kpts = gt_coords[:, -1, :]

        # Calculate difference between ground-truth coordinates
        # and estimated coordinates
        coord_error = est_coords[:, :-1, :] - gt_coords[:, :-1, :]
        coord_err = np.linalg.norm(coord_error, axis=1)

        # Count number of each keypoint
        kpt_count = np.sum(vis_kpts, axis=1)
        
        # Calculate percentage of correct keypoints
        less_than_threshold = (coord_err <= threshold) * vis_kpts
        PCK = 100. * np.sum(less_than_threshold, axis=1) / kpt_count
        PCK = np.ma.array(PCK, mask=False)
        kpt_count = np.ma.array(kpt_count, mask=False)
        kpt_ratio = kpt_count / np.sum(kpt_count).astype(np.float64)
        name_value = [('PCKh', np.sum(PCK * kpt_ratio))]
       
       ## PCK calculation check

        # Estimated keypoint coordinates
        pred = est_coords[:, :-1, :].transpose((2, 0, 1))

        # Ground-truth (GT) keypoint coordinates
        gt = gt_coords[:, :-1, :].transpose((2, 0, 1))

        # Mask indicating visible coordinates
        mask = (vis_kpts == 1).T

        # Modify GT keypoint coordinates
        gt[np.dstack((mask, mask)) == False] = -1

        # Image height and width
        H, W = self.ann_info['image_size']

        # Number of samples
        N = gt.shape[0]

        # PCK threshold
        thr = 0.05

        # Distance normalisation array
        normalize = np.tile(np.array([[H, W]]), (N, 1))

        # Calculate PCK using same code as during training
        acc, avg_acc, cnt = keypoint_pck_accuracy(pred, gt, mask, thr, normalize)

        # Check PCK calculations
        # assert round(name_value['PCKh']) == round(avg_acc * 100), 'There is a discrepancy in PCK value'

        # Update name_value
        name_value.append(('PCK', avg_acc * 100))
        
        # Convert into OrderedDict
        name_value = OrderedDict(name_value)

        if res_folder:
            # est_file = osp.join(res_folder, 'est_coords.mat')
            # savemat(est_file, mdict={'est_coords': est_coords})

            # Calculate PCK per image and mean keypoint loaction error per image
            distances = _calc_distances(pred, gt, mask, normalize)

            # Calculate PCK per image
            acc = np.array([_distance_acc(d, thr) for d in distances.T])

            # Replace -1 with NaN
            distances[distances == -1] = np.nan

            # Calculate number of visible keypoints
            vis_kpts = np.sum(np.isnan(distances)==False, axis=0)

            # Calculate mean keypoint location error per image
            mean_kle = np.nanmean(distances, axis=0)

            # Create pandas DataFrame
            df = pd.DataFrame({'VisKpts': vis_kpts, 'PCK': acc, 'MeanKLE': mean_kle})

            # Save df
            df.to_csv(osp.join(res_folder, 'per_image_accuracy.csv'), index=False)

        return name_value