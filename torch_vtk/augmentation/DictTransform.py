import math
import random
from abc import abstractmethod

import numpy as np
import torch
from torch.nn import functional as f

from torch_volume_dataloaders.augmentation.rotation_helper import RotationHelper


class DictTransform(object):

    def __init__(self, func, device="cpu", apply_on=["vol", "mask"], dtype=torch.float32, **kwargs):
        super().__init__()

        self.device = device
        self.apply_on = apply_on
        self.dtype = dtype

    @abstractmethod
    def transform(self, data): pass

    def __call__(self, data):
        for key in self.apply_on:
            data[key] = self.transform(data[key])
        return data


    def __call__(self, sample, **kwargs):
        # extract the volumes

        # todo dynamic keys.
        vol = sample["vol"]
        mask = sample["mask"]
        if self.device == "cuda":
            vol = vol.to(self.device)
            mask = mask.to(self.device)

        if self.apply_on == ["vol"]:
            tfms = self.func(self.device, **kwargs)
            vol = tfms(vol)
        else:
            # assert vol.shape == mask.shape
            sample = [vol, mask]
            tfms = self.func(self.device, **kwargs)
            # tfms = self.func()
            vol, mask = tfms(sample)

        # change dtype
        if self.dtype == torch.float16:
            vol = vol.to(torch.float16)
            if mask.dtype == torch.float32:
                mask = mask.to(torch.float16)

        return vol, mask


class RotateDictTransform(DictTransform):

    def __init__(self, device, **kwargs):
        """
            ,axis=0, fillcolor_vol=-1024, fillcolor_mask=0, degree=10
        :param args:
        :param axis:
        :param fillcolor_vol:
        :param fillcolor_mask:
        :param degree:
        """
        # transform to args
        # super(RotateDictTransform, self).__init__()
        self.apply_on = kwargs["apply_on"]
        DictTransform.__init__(self, func=RotateDictTransform, apply_on=self.apply_on, **kwargs)
        self.degree = kwargs["degree"]
        self.axis = kwargs["axis"]
        self.fillcolor_vol = kwargs["fillcolor_vol"]
        self.fillcolor_mask = kwargs["fillcolor_mask"]
        self.device = device
        self.rotation_helper = RotationHelper(self.device)

    @abstractmethod
    def __call__(self, sample):

        # if apply on both then ...
        if self.apply_on == ["vol"]:
            self.transform_vol(self, sample)
        else:
            self.transform_vol_mask(self, sample[0], sample[1])

    def transform_vol(self, vol):
        rotation_matrix = self.rotation_helper.get_rotation_matrix_random(1)
        # vol = vol.squeeze(0)
        if vol.dtype is torch.float16:
            vol = vol.to(torch.float32)
        vol = self.rotation_helper.rotate(vol, rotation_matrix)
        return vol

    def transform_vol_mask(self, vol, mask):
        # to vol and mask.
        rotation_matrix = RotationHelper.get_rotation_matrix_random(1)
        vol = RotationHelper.rotate(vol, rotation_matrix)
        mask = RotationHelper.rotate(mask, rotation_matrix)

        return [vol, mask]


class NoiseDictTransform(DictTransform):

    def __init__(self, device, **kwargs):
        """
        noise_variance=(0.001, 0.05)
        :param noise_variance:
        """
        # super(NoiseDictTransform, self).__init__(**kwargs)
        # todo super class
        self.noise_variance = kwargs["noise_variance"]
        self.device = device
        # DictTransform.__init__(self, **kwargs)

    def __call__(self, sample):
        variance = random.uniform(self.noise_variance[0], self.noise_variance[1])
        noise = np.random.normal(0.0, variance, size=sample.shape)

        noise = torch.from_numpy(noise)

        noise = noise.to(self.device)
        sample = torch.add(sample, noise)

        return sample


class BlurDictTransform(DictTransform):

    def __init__(self, device, **kwargs):
        """

        :param kwargs: muss contain channels kernel size and sigma
        """
        # super(BlurDictTransform, self).__init__(**kwargs)
        # DictTransform.__init__(self, **kwargs)
        self.channels = kwargs["channels"]
        self.sigma = [kwargs["sigma"], kwargs["sigma"], kwargs["sigma"]]
        self.kernel_size = kwargs["kernel_size"]
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
        self.weight = kernel
        self.groups = self.channels

        self.conv = f.conv3d


    def transform(self, data):
        if data.dtype == torch.float16:
            sample = data.to(torch.float32)
        sample = sample.unsqueeze(0)
        vol = self.conv(sample, weight=self.weight, groups=self.groups, padding=1)
        vol = vol.squeeze(0)
        return vol




class Cropping(DictTransform):
    def __init__(self, **kwargs):
        super(Cropping, self).__init__(**kwargs)
        DictTransform.__init__(self, **kwargs)

    def get_crop(self, t, min_i, max_i):
        ''' Crops `t` in the last 3 dimensions for given 3D `min_i` and `max_i` like t[..., min_i[j]:max_i[j],..]'''
        return t[..., min_i[0]:max_i[0], min_i[1]:max_i[1], min_i[2]:max_i[2]]

    def get_crop_around(self, t, mid, size):
        return t[mid[0] - size // 2:  mid[0] + size // 2,
               mid[1] - size // 2:  mid[1] + size // 2,
               mid[2] - size // 2:  mid[2] + size // 2]

    def get_center_crop(self, t, size):
        return self.get_crop_around(t, (torch.Tensor([*t.shape]) // 2).long(), size)
