#%%
import torch
from torch_volume_dataloaders.utils.volume_utils import make_nd, normalize_hounsfield
import numpy as np

import unittest
from tensor_testing_utils import TensorTestCase

# %%
class TestMakeND(TensorTestCase):
    def setUp(self):
        self.a   = torch.tensor([42])
        self.a2r = torch.tensor([[42]])
        self.a3r = torch.tensor([[[42]]])
        self.a4r = torch.tensor([[[[42]]]])
        self.a5r = torch.tensor([[[[[42]]]]])

    def test_values_match(self):
        self.assertSameValues(make_nd(self.a, 2), self.a2r)
        self.assertSameValues(make_nd(self.a, 3), self.a3r)
        self.assertSameValues(make_nd(self.a, 4), self.a4r)
        self.assertSameValues(make_nd(self.a, 5), self.a5r)

    def test_invalid_n_warning(self):
        with self.assertRaises(Exception): make_nd(self.a, 1)
        with self.assertRaises(Exception): make_nd(self.a, -42)
        with self.assertRaises(Exception): make_nd(self.a, 0)

    def test_shapes_match(self):
        self.assertSameShape(make_nd(self.a, 2), self.a2r)
        self.assertSameShape(make_nd(self.a, 3), self.a3r)
        self.assertSameShape(make_nd(self.a, 4), self.a4r)
        self.assertSameShape(make_nd(self.a, 5), self.a5r)

    def test_ndims_match(self):
        self.assertSameDims(make_nd(self.a, 2), self.a2r)
        self.assertSameDims(make_nd(self.a, 3), self.a3r)
        self.assertSameDims(make_nd(self.a, 4), self.a4r)
        self.assertSameDims(make_nd(self.a, 5), self.a5r)


class TestNormalizeHounsfield(TensorTestCase):
    def setUp(self):
        self.anp  = np.array([[-1, 0, 2],[500, 4095, 4096],[5000, -2e-3, -5e3]]).astype(np.float32)
        self.anpr = np.array([[0, 0, 2.0/4095],[500.0/4095, 1, 1],[1, 0, 0]]).astype(np.float32)
        self.apt  = torch.from_numpy(self.anp)
        self.aptr = torch.from_numpy(self.anpr)
        self.bpt  = torch.rand((1,1,10,10,10)) * 4200 - 50
        self.bnp  = self.bpt.numpy()

    def test_values_match(self):
        self.assertSameValues(normalize_hounsfield(self.anp), self.anpr)
        self.assertSameValues(normalize_hounsfield(self.apt), self.aptr)
        self.assertCloseValues(normalize_hounsfield((1,2,3)), np.array([0.0002442, 0.0004884, 0.0007326], dtype=np.float32))
        self.assertCloseValues(normalize_hounsfield([1,2,3]), np.array([0.0002442, 0.0004884, 0.0007326], dtype=np.float32))
        self.assertAlmostEqual(normalize_hounsfield(1), 0.0002442)


    def test_invalid_type_exception(self):
        with self.assertRaises(Exception): normalize_hounsfield('abcd')
        with self.assertRaises(Exception): normalize_hounsfield({'answer': 42})

    def test_boundaries(self):
        self.assertAll(normalize_hounsfield(self.bnp) >= 0.0)
        self.assertAll(normalize_hounsfield(self.bnp) <= 1.0)
        self.assertAll(normalize_hounsfield(self.bpt) >= 0.0)
        self.assertAll(normalize_hounsfield(self.bpt) <= 1.0)

    def test_nans(self):
        self.assertNotAny(   np.isnan(normalize_hounsfield(self.anp)))
        self.assertNotAny(torch.isnan(normalize_hounsfield(self.apt)))
        self.assertNotAny(   np.isnan(normalize_hounsfield(self.bnp)))
        self.assertNotAny(torch.isnan(normalize_hounsfield(self.bpt)))

    def test_shapes_match(self):
        self.assertSameShape(normalize_hounsfield(self.bnp), self.bnp)
        self.assertSameShape(normalize_hounsfield(self.bpt), self.bpt)
        self.assertSameDims(normalize_hounsfield(self.bnp),  self.bnp)
        self.assertSameDims(normalize_hounsfield(self.bpt),  self.bpt)

    def test_output_types(self):
        self.assertType(normalize_hounsfield(self.bnp.astype(np.float32)), np.float32)
        self.assertType(normalize_hounsfield(self.bpt, dtype=np.float32 ), np.float32)
        self.assertType(normalize_hounsfield(self.bpt.to(    torch.float32)), torch.float32)
        self.assertType(normalize_hounsfield(self.bnp, dtype=torch.float32 ), torch.float32)

if __name__ == '__main__':
    unittest.main()
