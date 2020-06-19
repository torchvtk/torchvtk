import unittest
import torch
import numpy as np

__all__ = ['TensorTestCase']

class TensorTestCase(unittest.TestCase):
    def assertSameValues(self, a, b): self.assertTrue((a==b).all())
    def assertCloseValues(self, a, b):
        if   torch.is_tensor(a) and torch.is_tensor(b):
            self.assertTrue(torch.allclose(a,b))
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            self.assertTrue(np.allclose(a,b))
        else: raise Exception(f'a ({a.dtype}) and b ({b.dtype}) must both be either torch.tensor or np.ndarray')
    def assertSameShape(self, a, b): self.assertEqual(a.shape, b.shape)
    def assertSameDims(self, a, b): self.assertEqual(a.ndim, b.ndim)
    def assertSameType(self, a, b): self.assertEqual(a.dtype, b.dtype)
    def assertShape(self, a, shape): self.assertEqual(a.shape, shape)
    def assertDims(self, a, ndims): self.assertEqual(a.ndims, ndims)
    def assertType(self, a, dtype): self.assertEqual(a.dtype, dtype)
    def assertAll(self, boolt): self.assertTrue(boolt.all())
    def assertNotAll(self, boolt): self.assertFalse(boolt.all())
    def assertAny(self, boolt): self.assertTrue(boolt.any())
    def assertNotAny(self, boolt): self.assertFalse(boolt.any())
