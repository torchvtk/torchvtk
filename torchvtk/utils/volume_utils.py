import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['make_nd', 'make_2d', 'make_3d', 'make_4d', 'make_5d',
    'normalize_hounsfield', 'normalize_voxel_scale']

def make_nd(t, n):
    '''  Prepends singleton dimensions to `t` until n-dimensional '''
    if n <= t.ndim:
        return t
    else:
        nons = [None]*(n-t.ndim)
        return t[nons]

def make_2d(t): return make_nd(t, 2)
def make_3d(t): return make_nd(t, 3)

def make_4d(t):
    '''  Prepends singleton dimensions to `t` until 4D '''
    return make_nd(t, 4)

def make_5d(t):
    '''  Prepends singleton dimensions to `t` until 5D '''
    return make_nd(t, 5)

def normalize_hounsfield(vol, dtype=None):
    ''' Normalizes `vol` by 4095 and clamps to [0,1]. `dtype=None` defaults to 32-bit float'''
    if   isinstance(dtype, torch.dtype): vol = torch.tensor(vol, dtype=dtype)
    elif np.issctype(dtype):             vol = np.array(    vol, dtype=dtype)
    elif dtype is None:
        vol = vol.float() if torch.is_tensor(vol) else np.array(vol).astype(np.float32)
    if   torch.is_tensor(vol):
        return torch.clamp(vol / 4095.0, 0.0, 1.0)
    elif isinstance(vol, np.ndarray):
        return np.clip(vol / 4095.0, 0.0, 1.0)
    else: raise Exception(f'vol (type={type(vol)}) is neither torch.tensor, nor np.ndarray')

def normalize_voxel_scale(vol, vox_scl):
    assert torch.is_tensor(vol)
    vox_scl = torch.tensor(vox_scl)
    assert vox_scl.shape == torch.Size([3])
    new_shape = torch.Size((torch.tensor(vol.shape) * vox_scl).round().long())
    return F.interpolate(make_5d(vol), size=new_shape, align_corners=True, mode='trilinear')
