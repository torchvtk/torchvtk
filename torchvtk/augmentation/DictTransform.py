import math
import random
from abc import abstractmethod
from torchvtk.utils.volume_utils import make_5d
import numpy as np
import torch
from torch.nn import functional as F


class DictTransform(object):

    def __init__(self, device="cpu", apply_on=["vol", "mask"], dtype=torch.float32, batch_transform=False):
        super().__init__()

        self.device = device
        self.apply_on = apply_on
        self.dtype = dtype
        self.batch_transform = batch_transform

    @abstractmethod
    def transform(self, data):
        pass

    def __call__(self, data):

        for key in self.apply_on:
            tmp = data[key]

            if not torch.is_tensor(tmp):
                tmp = torch.from_numpy(tmp)

            # to GPU.
            if self.device == "cuda":
                tmp = tmp.to(self.device)

            if tmp.dtype is torch.float16:
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


class NoiseDictTransform(DictTransform):

    def __init__(self, noise_variance=(0.001, 0.05), **kwargs):
        self.device = kwargs["device"]
        self.noise_variance = noise_variance
        DictTransform.__init__(self, **kwargs)

    def transform(self, data):
        std = torch.rand(data.size(0) if data.ndim > 5 else 1, device=data.device, dtype=data.dtype)
        return data + torch.randn_like(data) * std


class BlurDictTransform(DictTransform):

    def __init__(self, channels=1, kernel_size=(3, 3, 3), sigma=1, **kwargs):
        """

        :param device:
        :param channels:
        :param kernel_size:
        :param sigma:
        """
        self.channels = channels
        self.sigma = [sigma, sigma, sigma]
        self.kernel_size = kernel_size
        self.device = kwargs["device"]
        DictTransform.__init__(self, **kwargs)
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
        self.conv = F.conv3d

    def transform(self, data):
        vol = make_5d(data)
        vol = self.conv(vol, weight=self.weight, groups=self.channels, padding=1)
        return vol


class CroppingTransform(DictTransform):
    def __init__(self, size=20, position=0, **kwargs):
        DictTransform.__init__(self, **kwargs)
        self.size = size
        self.position = position

    def transform(self, data):
        data = self.get_center_crop(data, self.size)
        return data

    def get_crop_around(self, data, mid, size):
        size = size // 2
        if mid[0] == 0:
            mid = mid.squeeze(0)
        return data[..., mid[-3] - size:mid[-3] + size,
               mid[-2] - size:  mid[-2] + size,
               mid[-1] - size:  mid[-1] + size]

    def get_center_crop(self, data, size):
        t = torch.Tensor([*data.shape]) // 2
        return self.get_crop_around(data, (torch.Tensor([*data.shape]) // 2).long(), size)
