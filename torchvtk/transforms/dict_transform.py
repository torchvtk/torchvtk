import math
from abc import abstractmethod
from torchvtk.utils.volume_utils import make_5d
import numpy as np
import torch
from torch.nn import functional as F


######################
####  BASE CLASS #############################################################
#####################

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
        if isinstance(data, dict):
            for key in self.apply_on:
                tmp = data[key]
                if isinstance(tmp, np.ndarray):  # Convert from NumPy
                    tmp = torch.from_numpy(tmp)
                if torch.is_tensor(tmp):  # If tmp is tensor, control type and device
                    data[key] = self.transform(tmp.to(self.dtype).to(self.device))
                else:
                    data[key] = self.transform(tmp)
        elif isinstance(data, (list, tuple)):
            data = [self.transform(d) if torch.is_tensor(d) else d for d in data]
        elif torch.is_tensor(data):
            data = self.transform(data)
        else:
            raise Exception(f'Invalid data type for DictTransform: {type(data)}. Should be da dict (with keys to apply the transform on given through apply_on parameter), list, tuple or single tensor (applies to all tensors in these cases). Please modify your Dataset accordingly.')
        return data


#####################
#### Transforms ##############################################################
####################

class Lambda(DictTransform):
    def __init__(self, func, apply_on, **kwargs):
        ''' Applies a given function, wrapped in a `DictTransform`

        Args:
            func (function): The function to be executed
            apply_on (list of str): Dictionary keys to apply this function to in the sense of a `DictTransform`
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(apply_on=apply_on, **kwargs)
        self.tfm = func

    def transform(self, data): return self.tfm(data)

class Composite:
    def __init__(self, *tfms, apply_all_on=None):
        ''' Composites multiple transforms together
        Args:
            tfms (Callable, DictTransform): `DictTransform`s or just callable objects that can handle the incoming dict data
            apply_all_on (List of str): Overrides the `apply_on` dictionary masks of the given transforms. (Only applies to `DictTransform`s)
        '''
        self.tfms = [*tfms]
        if apply_all_on is not None:
            assert isinstance(apply_all_on, (list, tuple)) and isinstance(apply_all_on[0], str)
            for tfm in self.tfms:
                assert hasattr(tfm, '__call__')
                if isinstance(tfm, DictTransform):
                    tfm.apply_on = apply_all_on

    def __call__(self, data):
        for tfm in self.tfms:
            data = tfm(data)
        return data

    def __get__(self, i):
        return self.tfms[i]



class Resize(DictTransform):
    def __init__(self, size, mode='trilinear', **kwargs):
        ''' Resizes volumes to a given size or by a given factor

        Args:
            size (3-tuple/list or float): The new spatial dimensions in a tuple or a factor as scalar
            mode (str, optional): Resampling mode. See PyTorch's `torch.nn.functional.interpolate`. Defaults to 'trilinear'.
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        if isinstance(size, (list, tuple)):
            self.is_factor = False
        elif isinstance(size, (int, float)):
            self.is_factor = True
        else: raise Exception(f'Invalid size argument ({size}). Use 3-tuple/list for a target size of a float for resizing by a factor.')
        self.size = size
        self.mode = mode

    def transform(self, x):
        if self.is_factor:
            return F.interpolate(make_5d(x), scale=self.size, mode=self.mode).squeeze(0)
        else:
            return F.interpolate(make_5d(x), size=self.size, mode=self.mode).squeeze(0)


class GaussianNoise(DictTransform):
    """
    Transformations for adding noise to images.
    """

    def __init__(self, std_deviation=0.01, mean=0, **kwargs):
        """
        :param std_deviation: The variance of the noise added to the  image.
        :param mean: The mean of the noise.
        :param kwargs: Arguments for `DictTransform`.
        """
        self.std_deviation = std_deviation
        self.mean = mean
        DictTransform.__init__(self, **kwargs)

    def transform(self, data):
        """Applies the Noise onto the images. Variance is controlled by the noise_variance parameter."""
        min, max = data.min(), data.max()
        data = data + torch.randn_like(data) * self.std_deviation + self.mean
        return torch.clamp(data, min, max)


class GaussianBlur(DictTransform):
    """Transformation for adding Blur to images."""

    def __init__(self, channels=1, kernel_size=(3, 3, 3), sigma=1, **kwargs):
        """
        Initializing the Blur Transformation.
        :param channels: Amount of channels of the input data.
        :param kernel_size: Size of the convolution kernel.
        :param sigma: Standard deviation.
        :param kwargs: Arguments for `DictTransform`
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


class Crop(DictTransform):
    """
    Transformation for the cropping of 4D or 5D Volumes.
    """

    def __init__(self, size=(20,20,20), position=0, **kwargs):
        """
        :param size: Size of the crop.
        :param position: Middle point of the cropped region.
        :param kwargs: Arguments for `DictTransform`.
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
        return data[..., mid[-3] - size[0]: mid[-3] + size[0],
                         mid[-2] - size[1]: mid[-2] + size[1],
                         mid[-1] - size[2]: mid[-1] + size[2]]

    def get_center_crop(self, data, size):
        """Helper method for the crop."""
        if self.position == 0:
            return self.get_crop_around(data, (torch.Tensor([*data.shape]) // 2).long(), size)
        else:
            return self.get_crop_around(data, self.position, size)


class RandPermute(DictTransform):
    ''' Chooses one of the 8 random permutations for the volume axes '''
    def __init__(self, permutations=None, **kwargs):
        ''' Randomly choose one of the given permutations.
        Args:
            permutations (list of 3-tuples): Overrides the list of possible permutations to choose from.
                The default is  [ (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0) ].
                `permutations` must be a list or tuple of items that are compatible with torch.permute. Assume 0 to be the first spatial dimension, we account for a possible batch and channel dimension.
                The permutation will then be chosen at random from the given list/tuple.
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        if permutations is None:
            self.permutations = [ # All Possible permutations for volume
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0)
            ]
        else: self.permutations = permutations

    def transform(self, x):
        idx = torch.randint(0, len(self.permutations), (1,)) # Choose permutation
        pre_shap = tuple(range(x.ndim-3))
        post_shap = tuple(map(lambda d: d+len(pre_shap), self.permutations[idx]))
        return x.permute(*pre_shap, *post_shap).contiguous()


class RandFlip(DictTransform):
    ''' Flips dimensions with a given probability. (Random event occurs for each dimension)'''
    def __init__(self, flip_probability=0.5, dims=[1,1,1], **kwargs):
        ''' Flips dimensions of a tensor with a given `flip_probability`.
        Args:
            flip_probability (float): Probability of a dimension being flipped. Default 0.5.
            dims (list of 3 ints): Dimensions that may be flipped are denoted with a 1, otherwise 0. [1,0,1] would randomly flip a volumes depth and width dimension, while never flipping its height dimension
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        self.prob = flip_probability
        assert (isinstance(dims, (list, tuple)) or
            torch.is_tensor(dims) and dims.dtype == torch.long) and len(dims) == 3, "Invalid dims"
        self.dims = torch.LongTensor(dims)

    def transform(self, x):
        idxs = tuple((torch.nonzero(self.dims * torch.rand(3) < self.prob).view(-1) + x.ndim - 3).tolist())
        if len(idxs) == 0: return x
        return torch.flip(x, idxs).contiguous()
