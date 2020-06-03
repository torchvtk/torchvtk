import numpy as np
import torch
from torch import nn
from torch.nn import functional as f
import math
from PIL import Image
import random
from torch_volume_dataloaders.augmentation.rotation_helper import RotationHelper


class RotateDictTransform(object):

    def __init__(self, device=torch.device("cpu"), axis=0, fillcolor_vol=-1024, fillcolor_mask=0, degree=10,
                 apply_on=["vol", "mask"], dtype=torch.float32):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.degree = degree
        self.axis = axis
        self.fillcolor_vol = fillcolor_vol
        self.fillcolor_mask = fillcolor_mask
        self.apply_on = apply_on
        self.rotation_helper = RotationHelper(self.device)

    def __call__(self, sample):
        # dict
        if self.apply_on[0] is "vol":
            # to vol
            # get batch size and create random rotation matrix.
            print(len(sample["vol"]))
            rotation_matrix = self.rotation_helper.get_rotation_matrix_random(1)
            vol = sample["vol"]
            #vol = vol.squeeze(0)
            if vol.dtype is torch.float16:
                vol = vol.to(torch.float32)
            vol = self.rotation_helper.rotate(vol, rotation_matrix)
            # todo check if stacking works correctly.
            sample["vol"] = vol
        else:
            # to vol and mask.
            rotation_matrix = RotationHelper.get_rotation_matrix_random(len(sample))
            vol = sample["vol"]
            vol = RotationHelper.rotate(vol, rotation_matrix)
            sample["vol"] = vol
            mask = sample["mask"]
            mask = RotationHelper.rotate(mask, rotation_matrix)
            sample["mask"] = mask

        return sample


class NoiseDictTransform(object):

    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, noise_variance=(0.001, 0.05)):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.noise_variance = noise_variance

    def __call__(self, sample):
        vol = sample["vol"]
        
        variance = random.uniform(self.noise_variance[0], self.noise_variance[1])
        noise = np.random.normal(0.0, variance, size=vol.shape)
        
        if self.device == "cpu":
            # todo not sure if that works
            vol = vol + noise
        else:
            noise = torch.from_numpy(noise)

            #todo check if already on gpu for vol
            noise = noise.to(self.device)
            vol = vol.to(self.device)
            vol = torch.add(vol, noise)


        # conversion to the correct float type

        if self.dtype == torch.float16:
            vol = vol.to(torch.float16)
        else:
            vol = vol.to(torch.float32)
        sample["vol"] = vol
        return sample


class BlurDictTransform(object):

    def __init__(self, channels, kernel_size, device=torch.device("cpu"), dtype=torch.float32, sigma=10):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.device = device
        self.dtype = dtype
        self.sigma = [sigma, sigma, sigma]
        self.kernel_size = [kernel_size] * 3
        # code from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
        # initialize conv layer.
        kernel = 1

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
        vol = sample["vol"]

        if vol.dtype is torch.float16:
            vol = vol.to(torch.float32)

        if self.device is "cuda":
            vol = vol.to(self.device)
        vol = self.conv(vol, weight=self.weight, groups=self.groups, padding=1)
        sample["vol"] = vol
        return sample

