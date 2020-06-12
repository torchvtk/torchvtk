import math
import random

import numpy as np
import torch
from torch.nn import functional as f

from torch_volume_dataloaders.augmentation.rotation_helper import RotationHelper


class DictTransform(object):

    def __init__(self, func, device=torch.device("cpu"), apply_on=["vol", "mask"], dtype=torch.float32):
        super().__init__()
        self.func = func
        self.device = device
        self.apply_on = apply_on
        self.dtype = dtype

    def __call__(self, sample):
        # extract the volumes

        vol = sample["vol"]
        mask = sample["mask"]
        if self.device == "cuda":
            vol = vol.to(self.device)
            mask = mask.to(self.device)

        # do augmentations
        if self.apply_on == "vol":
            tfms = self.func()
            tfms(sample)
        else:
            sample = [vol, mask]
            tfms = self.func()
            tfms(sample)

        # change dtype
        if self.dtype == torch.float16:
            vol = vol.to(torch.float16)
            if mask.dtype == torch.float32:
                mask = mask.to(torch.float16)

        return vol, mask


class RotateDictTransform(DictTransform):

    def __init__(self, axis=0, fillcolor_vol=-1024, fillcolor_mask=0, degree=10):
        super().__init__()
        self.degree = degree
        self.axis = axis
        self.fillcolor_vol = fillcolor_vol
        self.fillcolor_mask = fillcolor_mask
        self.rotation_helper = RotationHelper(self.device)

    def transform_vol(self, vol):
        rotation_matrix = self.rotation_helper.get_rotation_matrix_random(1)
        # vol = vol.squeeze(0)
        if vol.dtype is torch.float16:
            vol = vol.to(torch.float32)
        vol = self.rotation_helper.rotate(vol, rotation_matrix)
        return vol


    def transform_vol__mask(self, vol, mask ):
        # to vol and mask.
        rotation_matrix = RotationHelper.get_rotation_matrix_random(len(sample))
        vol = RotationHelper.rotate(vol, rotation_matrix)
        mask = RotationHelper.rotate(mask, rotation_matrix)

        return [vol, mask]


class NoiseDictTransform(DictTransform):

    def __init__(self, noise_variance=(0.001, 0.05)):
        super().__init__()
        self.noise_variance = noise_variance

    def __call__(self, sample):
        variance = random.uniform(self.noise_variance[0], self.noise_variance[1])
        noise = np.random.normal(0.0, variance, size=sample.shape)

        noise = torch.from_numpy(noise)

        noise = noise.to(self.device)
        sample = torch.add(sample, noise)

        return sample


class BlurDictTransform(DictTransform):

    def __init__(self, channels, kernel_size, sigma=10):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = [sigma, sigma, sigma]
        self.kernel_size = [kernel_size] * 3
        # code from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
        # initialize conv layer.
        kernel = 1
        # todo add dynamic padding for higher kerner_size
        #
        self.meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, self.sigma, self.meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # todo check if that method of assigning works
        # self.register_buffer('weight', kernel)
        self.weight = kernel
        self.groups = channels

        self.conv = f.conv3d

    def __call__(self, sample):
        vol = self.conv(sample, weight=self.weight, groups=self.groups, padding=1)
        return vol


class Cropping(DictTransform):
    def __init__(self):
        super().__init__()

    def get_crop(self, t, min_i, max_i):
        ''' Crops `t` in the last 3 dimensions for given 3D `min_i` and `max_i` like t[..., min_i[j]:max_i[j],..]'''
        return t[..., min_i[0]:max_i[0], min_i[1]:max_i[1], min_i[2]:max_i[2]]

    def get_crop_around(self, t, mid, size):
        return t[mid[0] - size // 2:  mid[0] + size // 2,
               mid[1] - size // 2:  mid[1] + size // 2,
               mid[2] - size // 2:  mid[2] + size // 2]

    def get_center_crop(self, t, size):
        return self.get_crop_around(t, (torch.Tensor([*t.shape]) // 2).long(), size)

