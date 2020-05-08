import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['make_4D', 'make_5D']

def make_nd(t, n):
    '''  Prepends singleton dimensions to `t` until n-dimensional '''
    nons = [None]*(n-t.ndim)
    return t[nons]

def make_4d(t):
    '''  Prepends singleton dimensions to `t` until 5D '''
    return make_nd(t, 4)

def make_5d(t):
    '''  Prepends singleton dimensions to `t` until 5D '''
    return make_nd(t, 5)

def normalize_hounsfield(vol):
    ''' Normalizes `vol` by 4095 and clamps to [0,1] '''
    if   isinstance(vol, np.ndarray):
        return np.clip(vol / 4095.0, 0.0, 1.0)
    elif isinstance(vol, torch.tensor):
        return torch.clamp(vol / 4095.0, 0.0, 1.0)
