import math
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
            if isinstance(tmp, np.ndarray):  # Convert from NumPy
                tmp = torch.from_numpy(tmp)
            if torch.is_tensor(tmp):  # If tmp is tensor, control type and device
                data[key] = self.transform(tmp.to(self.dtype).to(self.device))
            else:
                data[key] = self.transform(tmp)
        return data


class NoiseDictTransform(DictTransform):
    """
    Transformations for adding noise to images.
    """

    def __init__(self, std_deviation=0.01, mean=0, **kwargs):
        """
        :param std_deviation: The variance of the noise added to the  image.
        :param mean: The mean of the noise.
        :param kwargs: Arguments of the super class.
        """
        self.std_deviation = std_deviation
        self.device = kwargs["device"]
        self.mean = mean
        DictTransform.__init__(self, **kwargs)

    def transform(self, data):
        """Applies the Noise onto the images. Variance is controlled by the noise_variance parameter."""
        min, max = data.min(), data.max()
        data = data + torch.randn_like(data) * self.std_deviation + self.mean
        return torch.clamp(data, min, max)


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
        self.pad = kernel_size[0] // 2

    def transform(self, data):
        """Applies the Blur using a 3D Convolution."""
        vol = F.pad(make_5d(data), (self.pad,) * 6, mode='replicate')
        return self.conv(vol, weight=self.weight, groups=self.channels, padding=0)


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
        if self.position != 0:
            try:
                assert (size[0]//2 <= position[0])
                assert (size[1]//2 <= position[1])
                assert (size[2]//2 <= position[2])
            except ValueError:
                print("The size is larger than the image allows on that center position.")


    def transform(self, data):
        "Applies the Center Crop."
        return self.get_center_crop(data, self.size)

    def get_crop_around(self, data, mid, size):
        """Helper method for the crop."""
        size = (size[0] // 2, size[1] // 2, size[2] // 2)
        if mid[0] == 0:
            mid = mid.squeeze(0)
        return data[..., mid[-3] - size[0]:mid[-3] + size[0],
               mid[-2] - size[1]:  mid[-2] + size[1],
               mid[-1] - size[2]:  mid[-1] + size[2]]

    def get_center_crop(self, data, size):
        """Helper method for the crop."""
        if self.position == 0:
            return self.get_crop_around(data, (torch.Tensor([*data.shape]) // 2).long(), size)

        else:
            return self.get_crop_around(data, self.position, size)
