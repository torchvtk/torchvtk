import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['make_4D', 'make_5D']

def make_4D(t):
    if t.ndim < 4: return make_4D(t[None])
    else:          return t

def make_5D(t):
    if t.ndim < 5: return make_5D(t[None])
    else:          return t
