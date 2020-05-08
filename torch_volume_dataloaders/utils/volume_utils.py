import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['make_4D', 'make_5D']

def make_nd(t, n):
    '''  Prepends singleton dimensions to `t` until n-dimensional '''
    if n <= t.ndim:
        raise Exception(f'make_nd expects n(={n}) to be larger than the current number of dimensions of t(={t.ndim}).')
    else:
        nons = [None]*(n-t.ndim)
        return t[nons]

def make_4d(t):
    '''  Prepends singleton dimensions to `t` until 5D '''
    return make_nd(t, 4)

def make_5d(t):
    '''  Prepends singleton dimensions to `t` until 5D '''
    return make_nd(t, 5)

def normalize_hounsfield(vol, dtype=None):
    ''' Normalizes `vol` by 4095 and clamps to [0,1]. `dtype` defaults to 32-bit float'''
    if isinstance(dtype, torch.dtype): vol = torch.tensor(vol, dtype=dtype)
    if np.issctype(dtype):             vol = np.array(    vol, dtype=dtype)
    if   torch.is_tensor(vol):
        return torch.clamp(vol / 4095.0, 0.0, 1.0)
    elif isinstance(vol, np.ndarray):
        return np.clip(vol / 4095.0, 0.0, 1.0)
    else: raise Exception(f'vol (type={type(vol)}) is neither torch.tensor, nor np.ndarray')
