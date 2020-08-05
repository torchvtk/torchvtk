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

### Basic Example
We have mechanisms to download some volume datasets automatically using torchvtk. These datasets are downloaded to your local torchvtk cache (`~/.torchvtk/` unless specified) where they are converted to `TorchDataset` format. If the dataset exists on your torchvtk folder, it will just be loaded.
```python
from torchvtk.datasets import TorchDataset

ds = TorchDataset.CQ500('custom/path/torchvtk', num_workers=4)
```
The `num_workers` parameter applies to downloading and unpacking.

### More involved example
This example showcases how `TorchDataset`s can be used to easily preprocess datasets, both for saving persisently on disk and during loading.
The following snippet shows how we can serialize a preprocessed version of the CQ500 dataset easily using multiprocessing:
```python
import torch.nn.functional as F
from torchvtk.datasets import TorchDataset
from torchvtk.utils    import make_5d

def to_256(data):
   data['vol'] = F.interpolate(make_5d(data['vol']), size=(256,256,256), mode='trilinear').squeeze(0)
   return data # Must return the dictionary!

ds = TorchDataset.CQ500('/mnt/hdd/torchvtk')
ds_256 = ds.cache_processed(to_256, 'CQ500_256', num_workers=4) # Call this only once
```
After calling this, the resized dataset is serialized. From then on use the following, assuming your local torchvtk folder is `/mnt/hdd/torchvtk/`:
```python
ds = TorchDataset('/mnt/hdd/torchvtk/CQ500_256')
```

Serializing large volume datasets at the resolution you will use for training (which is likely lower than the original) can be very beneficial for data loading times.

Having serialized the volumes in low resolution, we can apply more preprocessing in the `TorchDataset` that is applied upon loading a sample. This is ideal for `torchvtk.transforms`. We further split our data into train and validation:
```python
from torchvtk.transforms import GaussianNoise

train_ds = TorchDataset('/mnt/hdd/torchvtk/CQ500_256', filter_fn=lambda p: int(p.name[9:-3]) < 400,
   preprocess_fn=GaussianNoise(apply_on=['vol']))

valid_ds = TorchDataset('/mnt/hdd/torchvtk/CQ500_256', filter_fn=lambda p: int(p.name[9:-3]) >= 400)
```
We split our dataset into training and validation simply by using the `filter_fn` parameter which takes a function that filters out files from the dataset based on their filepath (`pathlib.Path`). Here the file's name is trimmed to the number specifically for the CQ500 item.
The `preprocess_fn` parameter takes any callable object and is expected to take a dictionary (as specified at the top of this article) and returns a modified dict.
`torchvtk.transforms` fulfill these requirements and can easily specify to which keys in your data the operations shall be applied. Check out [Transforms](Transforms.html).

### API
```eval_rst
.. autoclass:: torchvtk.datasets::TorchDataset
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
