# my_transforms.py
# Code to apply transformations to images with MMPose framework

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 1st September 2023

# Import required modules
from mmpose.datasets.builder import PIPELINES
import numpy as np
import cv2
from albumentations import Compose, OneOf, RandomRotate90, HorizontalFlip, VerticalFlip, KeypointParams


@PIPELINES.register_module()
class TopDownBBoxCrop:
    """Crop image according to bounding box and update keypoint coordinates accordingly.

    Required keys: 'bbox', 'ann_info', 'img'

    Modifies key: 'keypoints'

    Args:
        img_width (int): width of image in pixels.
            Default: 1920
        img_height (int): height of image in pixels.
            Default: 1200
        bbox_pc_inc (float): proportion by which box dimensions should be increased.
            Default: 0.05 (i.e. 5%)
        
    """

    def __init__(self,
                 img_width: int = 1920,
                 img_height: int = 1200,
                 bbox_pc_inc: float = 0.05):
        self.imw = img_width
        self.imh = img_height
        self.bbox_pc_inc = bbox_pc_inc

    def __call__(self, results):

        # Extract image name
        filename = results['image_file']
        
        # Extract bounding box information
        xmin, ymin, xmax, ymax = results['bbox']

        # Extract image
        image = results['img']

        # Extract keypoints
        keypoints = results['joints_3d']

        # Extract dimensions image should be resized to
        image_size = results['ann_info']['image_size']
        if image_size[0] == image_size[1]:
            image_size = image_size[0]

        # Box width and height
        w = abs(xmax - xmin)
        h = abs(ymax - ymin)
        
        # Adjust box width and height
        adj_w = self.bbox_pc_inc * w
        adj_h = self.bbox_pc_inc * h

        # If the bounding box width is greater or equal to the bounding box height
        if w >= h:
            
            # If the bounding box width is greater than the image height
            if w > self.imh:
                
                # Calculate amount of zero padding required
                h_adj = (w - self.imh)
                if (h_adj % 2) == 0:
                    h_lower = int(h_adj / 2)
                    h_upper = h_lower
                else:
                    h_lower = round(h_adj / 2)
                    h_upper = h_adj - h_lower

                # Zero-pad image
                image = cv2.copyMakeBorder(image, h_lower, h_upper, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # Update keypoint coordinates
                keypoints[:, 0] -= xmin
                keypoints[:, 1] += h_lower

                # Crop image
                if xmax == self.imw:
                    image = image[:, xmin:, :]
                else:
                    image = image[:, xmin:xmax, :]

                # Check image is square
                assert image.shape[0] == image.shape[1], f'Image {filename} cropped image is not square'

                # Resize image
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Update keypoint coordinates
                keypoints[:, 0] *= (image_size / w)
                keypoints[:, 1] *= (image_size / w)

            # If the bounding box width is equal to the image height
            elif w == self.imh:
                
                # Update keypoint coordinates
                keypoints[:, 0] -= xmin

                # Crop image
                if xmax == self.imw:
                    image = image[:, xmin:, :]
                else:
                    image = image[:, xmin:xmax, :]

                # Check image is square
                assert image.shape[0] == image.shape[1], f'Image {filename} cropped image is not square'

                # Resize image
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Update keypoint coordinates
                keypoints[:, 0] *= (image_size / w)
                keypoints[:, 1] *= (image_size / w)
            
            # If the bounding box is smaller than the image height
            else:
                
                # Calculate amount of bounding box height adjustment required
                h_adj = (w - h)
                if (h_adj % 2) == 0:
                    h_lower = int(h_adj / 2)
                    h_upper = h_lower
                else:
                    h_lower = round(h_adj / 2)
                    h_upper = h_adj - h_lower

                # Check that upper and lower adjustments are within range
                if (ymin - h_lower) >= 0:
                    if (ymax + h_upper) <= self.imh:
                        ymin -= h_lower
                        ymax += h_upper
                    else:
                        h_adj = (ymax + h_upper - self.imh)
                        if (ymin - h_lower - h_adj) >= 0:
                            h_lower += h_adj
                            ymin -= h_lower
                            ymax = self.imh
                        else:
                            print('error')
                else:
                    h_adj = abs(ymin - h_lower)
                    if (ymax + h_upper + h_adj) <= self.imh:
                        ymin = 0
                        ymax += (h_upper + h_adj)
                    else:
                        print('error')

                # Update keypoint coordinates
                keypoints[:, 0] -= xmin
                keypoints[:, 1] -= ymin

                # Crop image
                if xmax == self.imw:
                    if ymax == self.imh:
                        image = image[ymin:, xmin:, :]
                    else:
                        image = image[ymin:ymax, xmin:, :]
                else:
                    if ymax == self.imh:
                        image = image[ymin:, xmin:xmax, :]
                    else:
                        image = image[ymin:ymax, xmin:xmax, :]

                # Check image is square
                assert image.shape[0] == image.shape[1], f'Image {filename} cropped image is not square'

                # Resize image
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Update keypoint coordinates
                keypoints[:, 0] *= (image_size / w)
                keypoints[:, 1] *= (image_size / w)
                
        # If the bounding box width is smaller than the bounding box height
        else:
            
            # If the bounding box height is equal to the image height
            if self.imh == h:
                
                # Calculate amount of bounding box width adjustment required
                w_adj = h - w
                if (w_adj % 2) == 0:
                    w_left = int(w_adj / 2)
                    w_right = w_left
                else:
                    w_left = round(w_adj / 2)
                    w_right = w_adj - w_left

                # Check that right and left adjustments are within range
                if (xmin - w_left) >= 0:
                    if (xmax + w_right) <= self.imw:
                        xmin -= w_left
                        xmax += w_right
                    else:
                        w_adj = (xmax + w_right - self.imw)
                        if (xmin - w_left - w_adj) >= 0:
                            w_left += w_adj
                            xmin -= w_left
                            xmax = self.imw
                        else:
                            print('error')
                else:
                    w_adj = abs(xmin - w_left)
                    if (xmax + w_right + w_adj) <= self.imw:
                        xmin = 0
                        xmax += (w_right + w_adj)
                    else:
                        print('error')

                # Update keypoint coordinates
                keypoints[:, 0] -= xmin

                # Crop image
                if xmax == self.imw:
                    image = image[:, xmin:, :]
                else:
                    image = image[:, xmin:xmax, :]

                # Check image is square
                assert image.shape[0] == image.shape[1], f'Image {filename} cropped image is not square'

                # Resize image
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Update keypoint coordinates
                keypoints[:, 0] *= (image_size / h)
                keypoints[:, 1] *= (image_size / h)
            
            # If the bounding box height is smaller than the image height
            else:
                
                # Calculate amount of bounding box width adjustment required
                w_adj = h - w
                if (w_adj % 2) == 0:
                    w_left = int(w_adj / 2)
                    w_right = w_left
                else:
                    w_left = round(w_adj / 2)
                    w_right = w_adj - w_left

                # Check that right and left adjustments are within range
                if (xmin - w_left) >= 0:
                    if (xmax + w_right) <= self.imw:
                        xmin -= w_left
                        xmax += w_right
                    else:
                        w_adj = (xmax + w_right - self.imw)
                        if (xmin - w_left - w_adj) >= 0:
                            w_left += w_adj
                            xmin -= w_left
                            xmax = self.imw
                        else:
                            print('error')
                else:
                    w_adj = abs(xmin - w_left)
                    if (xmax + w_right + w_adj) <= self.imw:
                        xmin = 0
                        xmax += (w_right + w_adj)
                    else:
                        print('error')

                # Update keypoint coordinates
                keypoints[:, 0] -= xmin
                keypoints[:, 1] -= ymin

                # Crop image
                if xmax == self.imw:
                    if ymax == self.imh:
                        image = image[ymin:, xmin:, :]
                    else:
                        image = image[ymin:ymax, xmin:, :]
                else:
                    if ymax == self.imh:
                        image = image[ymin:, xmin:xmax, :]
                    else:
                        image = image[ymin:ymax, xmin:xmax, :]

                # Check image is square
                assert image.shape[0] == image.shape[1], f'Image {filename} cropped image is not square'

                # Resize image
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Update keypoint coordinates
                keypoints[:, 0] *= (image_size / h)
                keypoints[:, 1] *= (image_size / h)

        # Update results
        results['joints_3d'] = keypoints
        results['img'] = image
        results['scale'] = np.array([image_size / 200, image_size / 200])
        # NB division by 200 required because of scaling factor in transform_preds() 
        # in mmpose/core/post_processing/post_transforms.py
        results['center'] = np.array([(image_size - 1) / 2, (image_size - 1) / 2], dtype=np.float32)
        results['bbox'] = np.array([0, 0, image_size, image_size])
        
        return results


@PIPELINES.register_module()
class TopDownFliporRot90:
    """Data augmentation: flip image (horizontally or vertically) or rotate it by -90 or 90.

    Required key: 'img', 'joints_3d', and 'ann_info'.

    Modifies key: 'img', 'joints_3d'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Augment image by flippingation with image flip."""
        img = results['img']
        joints_3d = results['joints_3d']

        # Convert kpts to required format
        kpts = list(zip(list(joints_3d[:, 0]), list(joints_3d[:, 1])))

        # Albumentations transformation pipeline
        transform = Compose([
            OneOf([
                RandomRotate90(p=0.33),
                HorizontalFlip(p=0.33),
                VerticalFlip(p=0.33)
                ], p=1.00)
                ], p=0.75,
                keypoint_params=KeypointParams(format='xy', remove_invisible=False))
                    
        # Transform image and keypoints
        transformed = transform(image=img, keypoints=kpts)

        # Convert transformed keypoints to required format
        tr_kpts = np.zeros_like(joints_3d)
        tr_kpts[:, :2] = np.asarray(transformed['keypoints'])

        results['img'] = transformed['image']
        results['joints_3d'] = tr_kpts
        
        # Perform augmentation if criterion is met
        # if np.random.rand() <= self.flip_prob:

        #     # Either flip or rotate image
        #     rand_idx = np.random.randint(0, 4, 1)
        #     if rand_idx == 0: # Flip image left right
        #         img = img[:, ::-1, :]
        #         joints_3d[:, 0] = self.ann_info['img_size'][1] - joints_3d[:, 0]
        #     elif rand_idx == 1: # Flip image up down
        #         img = img[::-1, ...]
        #         joints_3d[:, 1] = self.ann_info['img_size'][0] - joints_3d[:, 1]
        #     elif rand_idx == 2: # Rotate image clockwise
        #         img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #         joints_3d
        #     else: # Rotate image anticlockwise
        #         img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #         joints_3d

        # results['img'] = img
        # results['joints_3d'] = joints_3d

        return results