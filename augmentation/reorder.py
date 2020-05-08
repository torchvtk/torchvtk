import torch
from torchvision.transforms import Compose, Lambda
from functools import partial

def RandPermute():
    permutations = [ # Possible permutations for volume
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0)
    ]
    def _tfm(x):
        idx = torch.randint(0, len(permutations), (1,)) # Choose permutation
        if isinstance(x, tuple): # apply to all volumes in tuple
            pre_shap = tuple(range(x[0].ndim-3)) # e.g. x of shape (8, 1, 64, 64, 64) -> (8, 1)
            post_shap = tuple(map(lambda d: d+len(pre_shap), permutations[idx])) # shift dim idxs
            return tuple(map(lambda t: t.permute(*pre_shap, *post_shap), x)) # permute all vols in tuple
        elif isinstance(x, list): # same as for tuple, but wrapped in list
            pre_shap = tuple(range(x[0].ndim-3))
            post_shap = tuple(map(lambda d: d+len(pre_shap), permutations[idx]))
            return list(map(lambda x: x.permute(*pre_shap, *post_shap), x))
        else: # assume x is a single tensor
            pre_shap = tuple(range(x.ndim-3))
            post_shap = tuple(map(lambda d: d+len(pre_shap), permutations[idx]))
            return x.permute(*pre_shap, *post_shap)
    return _tfm

def RandFlip():
    def _tfm(x):
        idxs = tuple(torch.nonzero(torch.rand(3) < 0.5).view(-1).tolist())
        if len(idxs) == 0: return x
        if isinstance(x, tuple):
            return tuple(map(partial(torch.flip, dims=idxs), x))
        elif isinstance(x, list):
            return list(map(partial(torch.flip, dims=idxs), x))
        else: return torch.flip(x, idxs)
    return _tfm
