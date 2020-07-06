import math
import random
from abc import abstractmethod
from torchvtk.utils.volume_utils import make_5d
import numpy as np
import torch
from torch.nn import functional as F


class DictTransform(object):
    """
    Super Class for the Transforms.


    """

    def __init__(self, device="cpu", apply_on=["vol", "mask"], dtype=torch.float32):
        """
        :param device: The torch Code for the device on which the transformation should be executed. Possiblities are ["cpu", "cuda"].
        :param apply_on: The keys of the volumes that should be transformed.
        :param dtype: The torch type in which the data are. Possibilities are [torch.float16, torch.float32].
        """
        super().__init__()

        self.device = device
        self.apply_on = apply_on
        self.dtype = dtype

    @abstractmethod
    def transform(self, data):
        """ Transformation Method, must be overwritten by every SubClass."""
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
            data[key] = tmp
        return data


class NoiseDictTransform(DictTransform):
    """
    Transformations for adding noise to images.
    """

    def __init__(self, noise_variance=(0.001, 0.05), **kwargs):
        """

        :param noise_variance: The variance of the noise added to the  image.
        :param kwargs: Arguments of the super class.
        """
        self.device = kwargs["device"]
        self.noise_variance = noise_variance
        DictTransform.__init__(self, **kwargs)

    def transform(self, data):
        """Applies the Noise onto the images. Variance is controlled by the noise_variance parameter."""
        std = torch.rand(data.size(0) if data.ndim > 5 else 1, device=data.device, dtype=data.dtype)
        return data + torch.randn_like(data) * std


class BlurDictTransform(DictTransform):
    """Transformation for adding Blur to images."""

    def __init__(self, channels=1, kernel_size=(3, 3, 3), sigma=1, **kwargs):
        """
        Initializing the Blur Transformation.
        :param channels: Amount of channels of the input data.
        :param kernel_size: Size of the convolution kernel.
        :param sigma: Standard deviation.
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

        kernel = kernel.to(self.device)
        self.weight = kernel
        self.conv = F.conv3d

    def transform(self, data):
        """Applies the Blur using a 3D Convolution."""
        vol = make_5d(data)
        vol = F.pad(make_5d(data), (self.pad, )*6, mode='replicate')
        return self.conv(vol, weight=self.weight, groups=self.channels, padding=0)
        return vol


class CroppingTransform(DictTransform):
    """
    Transformation for the cropping of 4D or 5D Volumes.
    """

    def __init__(self, size=20, position=0, **kwargs):
        """
        :param size: Size of the crop.
        :param position: Middle point of the cropped region.
        :param kwargs: Arguments for super class.
        """
        DictTransform.__init__(self, **kwargs)
        self.size = size
        self.position = position

    def transform(self, data):
        "Applies the Center Crop."
        return self.get_center_crop(data, self.size)

    def get_crop_around(self, data, mid, size):
        """Helper method for the crop."""
        size = size // 2
        if mid[0] == 0:
            mid = mid.squeeze(0)
        return data[..., mid[-3] - size:mid[-3] + size,
               mid[-2] - size:  mid[-2] + size,
               mid[-1] - size:  mid[-1] + size]

    def get_center_crop(self, data, size):
        """Helper method for the crop."""
        t = torch.Tensor([*data.shape]) // 2
        return self.get_crop_around(data, (torch.Tensor([*data.shape]) // 2).long(), size)
