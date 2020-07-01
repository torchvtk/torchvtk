## TorchDataset

This is your go-to dataset class to handle large volumes.
Each volume is stored together as a serialized binary PyTorch file (`.pt`), together with its meta data. You can always load such a file like this:
```python
import torch
data = torch.load('/path/to/file.pt´)
```
In the case of an item from the [CQ500](CQ500.md) datasets, data would be a dictionary of the form
```python
{
   'vol': torch.Tensor(),     # Volume tensor of shape (C, D, H, W)
   'vox_scl': torch.Tensor(), # Scale of the voxels (for rectangular grids)
   'name': str                # Name of the volume
}
```
Note that volumes saved in the torchvtk format are always scaled to `[0,1]` and saved as `torch.float16` or `torch.float32`. Also the shape for single-channel volumes is always 4-dimensional: `(1, D, H, W)`.

```eval_rst
.. autoclass:: torchvtk.datasets.TorchDataset
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
