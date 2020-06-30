import math
import random
from abc import abstractmethod

import numpy as np
import torch
from torch.nn import functional as f

from torchvtk.augmentation.rotation_helper import RotationHelper


class DictTransform(object):

    def __init__(self, device="cpu", apply_on=["vol", "mask"], dtype=torch.float32):
        super().__init__()

        self.device = device
        self.apply_on = apply_on
        self.dtype = dtype

    @abstractmethod
    def transform(self, data): pass

    def __call__(self, data):

        for key in self.apply_on:
            tmp = data[key]

            if not torch.is_tensor(tmp):
                tmp = torch.from_numpy(tmp)

            if self.device == "cuda":
                tmp = tmp.to(self.device)

            if self.dtype is torch.float16:
                tmp = tmp.to(torch.float32)

            tmp = self.transform(tmp)

            if self.dtype is torch.float16:
                tmp = tmp.to(torch.float16)
            elif self.dtype is not torch.float16 or torch.float32:
                tmp = tmp.to(torch.float32)
            tmp = tmp.to(self.device)
            # back on device?
            data[key] = tmp
        return data


class RotateDictTransform(DictTransform):

    def __init__(self, device, degree=4, axis=0, apply_on=["vol"], fillcolor_vol=0, fillcolor_mask=0, dtype=torch.float32):
        """
            ,axis=0, fillcolor_vol=-1024, fillcolor_mask=0, degree=10
        :param args:
        :param axis:
        :param fillcolor_vol:
        :param fillcolor_mask:
        :param degree:
        """
        # transform to args
        DictTransform.__init__(self, device, apply_on=apply_on, dtype=dtype)
        self.degree = degree
        self.axis = axis
        self.fillcolor_vol = fillcolor_vol
        self.fillcolor_mask = fillcolor_mask
        self.rotation_helper = RotationHelper(device)
        self.rotation_matrix = None

    def transform(self, data):
        if self.rotation_matrix is None:
            rotation_matrix = self.rotation_helper.get_rotation_matrix_random(1)
        data = self.rotation_helper.rotate(data, rotation_matrix)
        return data


class NoiseDictTransform(DictTransform):

    def __init__(self, device, noise_variance=(0.001, 0.05), apply_on=["vol"], dtype=torch.float32):

        self.device = device
        self.noise_variance = noise_variance
        DictTransform.__init__(self, self.device, apply_on=apply_on, dtype=dtype)

    def transform(self, data):
        #todo allow for batch
        variance = random.uniform(self.noise_variance[0], self.noise_variance[1])
        noise = np.random.normal(0.0, variance, size=data.shape)
        noise = torch.from_numpy(noise)
        noise = noise.to(self.device)
        sample = torch.add(data, noise)
        return sample


class BlurDictTransform(DictTransform):

    def __init__(self, apply_on=["vol"], dtype=torch.float32, device= "cpu", channels=1, kernel_size=(3, 3, 3), sigma=1):
        """

        :param device:
        :param channels:
        :param kernel_size:
        :param sigma:
        """
        self.channels = channels
        self.sigma = [sigma, sigma, sigma]
        self.kernel_size = kernel_size
        self.device = device
        DictTransform.__init__(self, self.device, apply_on=apply_on, dtype=dtype)
        # code from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
        # initialize conv layer.
        kernel = 1
        # todo add dynamic padding for higher kerner_size
        #
        self.meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in self.kernel_size
            ]
        )
        for size, std, mgrid in zip(self.kernel_size, self.sigma, self.meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.channels, *[1] * (kernel.dim() - 1))

        # todo check if that method of assigning works
        # self.register_buffer('weight', kernel)
        kernel = kernel.to(self.device)
        self.weight = kernel
        self.groups = self.channels

        self.conv = f.conv3d

    def transform(self, data):
        sample = data.unsqueeze(0)
        vol = self.conv(sample, weight=self.weight, groups=self.groups, padding=1)
        vol = vol.squeeze(0)
        return vol


class CroppingTransform(DictTransform):
    def __init__(self, apply_on=["vol"], dtype=torch.float32, device= "cpu", size=20, position=0):
        DictTransform.__init__(self, device=device, apply_on=apply_on, dtype=dtype)
        self.size = size
        self.position = position


    def transform(self, data):
        data= self.get_center_crop(data, self.size)
        return data

    def get_crop(self, t, min_i, max_i):
        ''' Crops `t` in the last 3 dimensions for given 3D `min_i` and `max_i` like t[..., min_i[j]:max_i[j],..]'''
        return t[..., min_i[0]:max_i[0], min_i[1]:max_i[1], min_i[2]:max_i[2]]

    # fixme wrong dimensions get cropped
    def get_crop_around(self, data, mid, size):
        size = size // 2
        if mid[0] == 0:
            mid = mid.squeeze(0)
        return data[...,mid[-3] - size:mid[-3] + size,
               mid[-2] - size:  mid[-2] + size,
               mid[-1] - size:  mid[-1] + size]

    def get_center_crop(self, data, size):
        t = torch.Tensor([*data.shape]) // 2
        return self.get_crop_around(data, (torch.Tensor([*data.shape]) // 2).long(), size)
