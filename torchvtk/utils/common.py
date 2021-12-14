import torch
import numpy as np

def clone(val):
    ''' Clone `val` if it is np.array or torch.Tensor. '''
    if   isinstance(val, np.ndarray):          return np.copy(val)
    elif torch.is_tensor(val):                 return torch.clone(val)
    elif isinstance(val, (list, dict, tuple)): return val.copy()
    else:                                      return val
