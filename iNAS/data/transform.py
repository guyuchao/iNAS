import cv2
import math
import numpy as np
import random
import torch

# salient object detection


class Resize(object):

    def __init__(self, base_size, image_only=False):
        self.h, self.w = base_size
        self.image_only = image_only

    def __call__(self, data_dict):
        image = data_dict['image']
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        data_dict['image'] = image

        if not self.image_only:
            label = data_dict['label']
            label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            data_dict['label'] = label

        return data_dict


class ToTensor(object):

    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.), image_only=False, label_normalize=True):
        self.mean = mean
        self.std = std
        self.image_only = image_only
        self.label_normalize = label_normalize

    def __call__(self, data_dict):
        image = data_dict['image']
        image = image.transpose(2, 0, 1).astype(np.float32)
        image = torch.from_numpy(image).div_(255)
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        image = image.sub(mean).div(std)
        data_dict['image'] = image

        if not self.image_only:
            label = data_dict['label']
            if self.label_normalize:
                label = torch.from_numpy((label / 255).astype(np.int64))  # 2-class sod 0/1
            else:
                label = torch.from_numpy(label.astype(np.int64))
            data_dict['label'] = label
        return data_dict


class RandomHorizontalFlip(object):

    def __init__(self, image_only=False):
        self.image_only = image_only

    def __call__(self, data_dict):
        if random.random() < 0.5:
            image = data_dict['image']
            image = cv2.flip(image, 1)
            data_dict['image'] = image
            if not self.image_only:
                label = data_dict['label']
                label = cv2.flip(label, 1)
                data_dict['label'] = label
        return data_dict


class Identity(object):

    def __call__(self, data_dict):
        return data_dict


# semantic segmentation


class RandomResizedCrop(object):

    def __init__(self, scales=(0.5, 1.), size=(512, 1024)):
        self.scales = scales
        self.size = size

    def __call__(self, data_dict):

        image, label = data_dict['image'], data_dict['label']
        assert image.shape[:2] == label.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        image_h, image_w = [math.ceil(size * scale) for size in image.shape[:2]]
        image = cv2.resize(image, (image_w, image_h))
        label = cv2.resize(label, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        if (image_h, image_w) == (crop_h, crop_w):
            data_dict['image'] = image
            data_dict['label'] = label
            return data_dict

        pad_h, pad_w = 0, 0
        if image_h < crop_h:
            pad_h = (crop_h - image_h) // 2 + 1
        if image_w < crop_w:
            pad_w = (crop_w - image_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
            label = np.pad(label, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        image_h, image_w, _ = image.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (image_h - crop_h)), int(sw * (image_w - crop_w))
        image = image[sh:sh + crop_h, sw:sw + crop_w, :].copy()
        label = label[sh:sh + crop_h, sw:sw + crop_w].copy()
        data_dict['image'] = image
        data_dict['label'] = label
        return data_dict
