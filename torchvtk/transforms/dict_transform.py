# %%
import math, random
from itertools import combinations
from abc import abstractmethod
from torchvtk.utils.volume_utils import make_nd
from torchvtk.utils import clone
import numpy as np
import torch
from torch.nn import functional as F


######################
####  BASE CLASS #############################################################
#####################

class DictTransform:
    """
    Super Class for the Transforms.
    """

    def __init__(self, device=None, apply_on=None, dtype=None):
        """
        :param apply_on: The keys of the item dictionaries on which the transform should be applied. Defaults to applying to all torch.Tensors
        :param device: The torch.device on which the transformation should be executed. Also valid: "cpu", "cuda". Defaults to using whatever comes.
        :param dtype: The torch.dtype to which the data should be converted before the transform. Defaults to using whatever comes..
        """
        if device is not None:
            assert (isinstance(device, torch.device)
                 or device in ['cpu', 'cuda']
                 or device.startswith('cuda:'))
        if apply_on is not None:
            assert isinstance(apply_on, (list, tuple, str))
            if isinstance(apply_on, (list, tuple)):
                assert len(apply_on) > 0 and isinstance(apply_on[0], str)
            else:
                apply_on = [apply_on]
        if dtype is not None:
            assert isinstance(dtype, torch.dtype)
        self.device = device
        self.apply_on = apply_on
        self.dtype = dtype

    @abstractmethod
    def transform(self, data):
        """ Transformation Method, must be overwritten by every SubClass."""
        pass

    def override_apply_on(self, apply_on):
        if self.apply_on is None:
            if isinstance(apply_on, (list, tuple)) and isinstance(apply_on[0], str):
                self.apply_on = apply_on
            elif isinstance(apply_on, str):
                self.apply_on = [apply_on]

    def __call__(self, inp):
        data = clone(inp)
        def to_tensor(tmp):
            if isinstance(tmp, np.ndarray):  # Convert from NumPy
                tmp = torch.from_numpy(tmp)
            if torch.is_tensor(tmp): return tmp.to(self.dtype).to(self.device)
            else:                    return tmp

        if isinstance(data, dict):
            if self.apply_on is None:
                keys, _ = zip(*filter(lambda tup: torch.is_tensor(tup[1]), data.items()))
            else: keys = self.apply_on

            for key, res in zip(keys, self.transform(list(map(to_tensor, [data[key] for key in keys])))):
                data[key] = res

        elif isinstance(data, (list, tuple)):
            data = self.transform(list(map(to_tensor, data)))
        elif torch.is_tensor(data):
            data = self.transform([data.clone()])
        elif isinstance(data, np.ndarray):
            data = self.transform([to_tensor(data)])
        else:
            raise Exception(f'Invalid data type for DictTransform: {type(data)}. Should be da dict (with keys to apply the transform on given through apply_on parameter), list, tuple or single tensor (applies to all tensors in these cases). Please modify your Dataset accordingly.')
        return data


#####################
#### Transforms ##############################################################
####################

class Lambda(DictTransform):
    def __init__(self, func, as_list=False, **kwargs):
        ''' Applies a given function, wrapped in a `DictTransform`

        Args:
            func (function): The function to be executed
            as_list (bool): Wether all inputs specified in `apply_on` are passed as a list, or as separate items. Defaults to False (separate items).
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        self.as_list = as_list
        self.tfm = func

    def transform(self, items):
        if self.as_list:
            return self.tfm(items)
        else:
            return [self.tfm(x) for x in items]

class RandLambda(DictTransform):
    def __init__(self, func, rand_range=(0,1), as_list=False, **kwargs):
        ''' Applies a given function, wrapped in a `DictTransform`

        Args:
            func (function): The function to be executed
            as_list (bool): Wether all inputs specified in `apply_on` are passed as a list, or as separate items. Defaults to False (separate items).
            rand_range (2-tuple): Min and Max for drawing a random variable uniformly. Defaults to (0,1)
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        self.as_list = as_list
        self.mi, self.ma = rand_range
        self.tfm = func

    def transform(self, items):
        r = np.random.uniform(self.mi, self.ma, (1,)).item()
        if self.as_list:
            return self.tfm(items, r)
        else:
            return [self.tfm(x, r) for x in items]

class Composite(DictTransform):
    def __init__(self, *tfms, apply_on=None, device=None, dtype=None):
        ''' Composites multiple transforms together

        Args:
            tfms (Callable, DictTransform): `DictTransform`s or just callable objects that can handle the incoming dict data
            apply_on (List of str): Overrides the `apply_on` dictionary masks of the given transforms. (Only applies to `DictTransform`s)
            device (torch.device, str): torch.device, `'cpu'` or `'cuda'`. This overrides the device for all `DictTransform`s.
            dtype (torch.dtype): Overrides the dtype for all `DictTransform`s this composites.
        '''
        super().__init__()
        self.tfms = [*tfms]
        if apply_on is not None:
            for tfm in self.tfms:
                assert hasattr(tfm, '__call__')
                if isinstance(tfm, DictTransform):
                    tfm.override_apply_on(apply_on)
                    if tfm.device   is None: tfm.device   = device
                    if tfm.dtype    is None: tfm.dtype    = dtype

    def __call__(self, data):
        for tfm in self.tfms:
            data = tfm(data)
        return data

    def __get__(self, i):
        return self.tfms[i]



class Resize(DictTransform):
    def __init__(self, size, mode='trilinear', is_batch=False, **kwargs):
        ''' Resizes volumes to a given size or by a given factor

        Args:
            size (tuple/list or float): The new spatial dimensions in a tuple or a factor as scalar
            mode (str, optional): Resampling mode. See PyTorch's `torch.nn.functional.interpolate`. Defaults to 'trilinear'.
            is_batch (bool): Wether the data passed in here already has a batch dimension (cannot be inferred if `size` is given as scalar). Defaults to False.
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
        self.is_batch = is_batch

    def transform(self, items):
        def tfm(x):
            if self.is_factor:
                if self.is_batch:
                    return F.interpolate(x, mode=self.mode, scale_factor=self.size)
                else:
                    return F.interpolate(x[None], mode=self.mode, scale_factor=self.size).squeeze(0)
            else:
                return F.interpolate(make_nd(x, len(self.size)+2), mode=self.mode, size=self.size).squeeze(0)
        return [tfm(x) for x in items]


class Noop(DictTransform):
    def __init__(self, **kwargs):
        ''' Just sets device and dtype, does nothing else.

        Args:
            device (str, torch.device): The device to move the tensors on. Defaults to not changing anything
            dtype (torch.dtype): The dtype to move the device to. Defaults to not changing anything
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)

    def transform(self, x): return x

class NormalizeMinMax(DictTransform):
    def __init__(self, min=0.0, max=1.0, **kwargs):
        ''' Normalizes tensors to a set min-max range

        Args:
            min (float, optional): New minimum value. Defaults to 0.0.
            max (float, optional): New maximum value. Defaults to 1.0.
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        self.min = min
        self.max = max

    def transform(self, items):
        def tfm(x):
            mi, ma = x.min(), x.max()
            scl = (self.max - self.min) / (ma - mi)
            return (x - mi) * scl + self.min
        return [tfm(x) for x in items]

class NormalizeStandardize(DictTransform):
    def __init__(self, mean=0.0, std=1.0, **kwargs):
        ''' Normalizes tensors to have a set mean and standard deviation

        Args:
            mean (float, optional): New mean of the sample. Defaults to 0.0.
            std (float, optional): New standard deviation of the sample. Defaults to 1.0.
            kwargs: Arguments for `DictTransform`
        '''
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def transform(self, items):
        def tfm(x):
            mean, std = x.mean(), x.std()
            scl = self.std / std
            return (x - mean) * scl + self.mean
        return [tfm(x) for x in items]
class GaussianNoise(DictTransform):
    def __init__(self, std_deviation=0.01, mean=0, **kwargs):
        """ Adds Gaussian noise to tensors

        Args:
            std_deviation (float, tensor): The variance of the noise
            mean (float, tensor): The mean of the noise.
            kwargs: Arguments for `DictTransform`.
        """
        self.std_deviation = std_deviation
        self.mean = mean
        DictTransform.__init__(self, **kwargs)

    def transform(self, items):
        """Applies the Noise onto the images. Variance is controlled by the noise_variance parameter."""
        noise = torch.randn_like(items[0]) * self.std_deviation + self.mean
        def tfm(x):
            mi, ma = x.min(), x.max()
            return torch.clamp(x + noise, mi, ma)
        return [tfm(x) for x in items]


class GaussianBlur(DictTransform):
    def __init__(self, channels=1, kernel_size=(3, 3, 3), sigma=1, **kwargs):
        """ Blurs tensors using a Gaussian filter

        Args:
            channels (int): Amount of channels of the input data.
            kernel_size (list of int): Size of the convolution kernel.
            sigma (float): Standard deviation.
            kwargs: Arguments for `DictTransform`
        """
        self.channels = channels
        self.sigma = [sigma, sigma, sigma]
        self.kernel_size = kernel_size
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

        self.weight = kernel
        self.conv = F.conv3d
        self.pad = kernel_size[0] // 2

    def transform(self, items):
        """Applies the Blur using a 3D Convolution."""
        def tfm(x):
            vol = F.pad(make_nd(x, len(self.kernel_size)+2), (self.pad,) * 2*len(self.kernel_size), mode='replicate')
            return self.conv(vol, weight=self.weight.to(x.dtype).to(x.device), groups=self.channels, padding=0)
        return [tfm(x) for x in items]


class Crop(DictTransform):
    def __init__(self, size=(20,20,20), position=0, **kwargs):
        """ Crops a tensor
            size (3-tuple of int): Size of the crop.
            position (3-tuple of int): Middle point of the cropped region.
            kwargs: Arguments for `DictTransform`.
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

    def transform(self, items):
        "Applies the Center Crop."
        return [self.get_center_crop(x, self.size) for x in items]

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


class RandCrop(DictTransform):
    def __init__(self, tile_sz, dim=3, **kwargs):
        super().__init__(**kwargs)
        if isinstance(tile_sz, (tuple, list)):
            self.tile_sz = torch.LongTensor(tile_sz)
        else:
            self.tile_sz = torch.LongTensor([tile_sz] * dim)
        self.dim = dim
    def transform(self, items):
        shap = torch.LongTensor(list(items[0].shape[-self.dim:]))
        begin = torch.floor(torch.rand((self.dim,)) * (shap - self.tile_sz)).long()
        end = begin + self.tile_sz
        idx = [slice(None, None)]*(items[0].ndim-self.dim) + [slice(b, e) for b,e in zip(begin.tolist(), end.tolist())]
        return [clone(item[idx]) for item in items]

class RandCropResize(DictTransform):
    def __init__(self, min_tile_sz, dim=3, **kwargs):
        super().__init__(**kwargs)
        if isinstance(min_tile_sz, (tuple, list)):
            self.min_tile_sz = torch.LongTensor(min_tile_sz)
        else:
            self.min_tile_sz = torch.LongTensor([min_tile_sz] * dim)
        self.dim = dim
    def transform(self, items):
        shap = torch.LongTensor(list(items[0].shape[-self.dim:]))
        tile_sz = torch.randint(self.min_tile_sz[-1], shap.min().item(), (1,)).expand(self.dim)
        begin = torch.floor(torch.rand((self.dim,)) * (shap - tile_sz)).long()
        end = begin + tile_sz
        idx = [slice(None, None)]*(items[0].ndim-self.dim) + [slice(b, e) for b,e in zip(begin.tolist(), end.tolist())]
        print(idx)
        print(tile_sz.tolist())
        return [F.interpolate(make_nd(item[idx], self.dim+2), size=self.min_tile_sz.tolist(), mode='trilinear').squeeze(0) for item in items]

class RandPermute(DictTransform):
    ''' Chooses one of the 8 random permutations for the volume axes '''
    def __init__(self, permutations=None, **kwargs):
        ''' Randomly choose one of the given permutations.

        Args:
            permutations (list of 3-tuples): Overrides the list of possible permutations to choose from. The default is  [ (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0) ]. `permutations` must be a list or tuple of items that are compatible with torch.permute. Assume 0 to be the first spatial dimension, we account for a possible batch and channel dimension. The permutation will then be chosen at random from the given list/tuple.
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

    def transform(self, items):
        idx = torch.randint(0, len(self.permutations), (1,)) # Choose permutation
        pre_shap = tuple(range(items[0].ndim-3))
        post_shap = tuple(map(lambda d: d+len(pre_shap), self.permutations[idx]))
        return [x.permute(*pre_shap, *post_shap).contiguous() for x in items]


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

    def transform(self, items):
        idxs = tuple((torch.nonzero(self.dims * torch.rand(3) < self.prob, as_tuple=False).view(-1) + items[0].ndim - 3).tolist())
        if len(idxs) == 0: return items
        return [torch.flip(x, idxs).contiguous() for x in items]

class RandRot90(DictTransform):
    ''' Randomly rotates between 0-3 times for 90 degrees along each axis in random order '''
    def __init__(self, ndim, **kwargs):
        ''' Randomly rotates tensor.

        Args:
            ndim (int): Number of spatial dimensions to be rotated.ArithmeticError
        '''
        super().__init__(**kwargs)
        self.ndim = ndim

    def transform(self, items):
        n = torch.randint(0, 4, (self.ndim,))
        axs = list(combinations(range(-self.ndim,0), 2))
        random.shuffle(axs)
        def tfm(x):
            for k, ax in zip(n, axs):
                x = torch.rot90(x, k, ax)
            return x
        return [tfm(x) for x in items]
# %%
