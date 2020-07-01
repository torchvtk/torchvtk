## torchvtk Converters
Converters take care of reading existing public datasets from disk in their original form and convert them to torchvtk's default form, which is `TorchDataset`.

Many of the scripts can be used directly as a command line tools.

### Examples
#### Convert the QureAI CQ500 dataset to TorchDataset
If you don't already have the data, we recommend using our dataset utils to get the dataset easily:
```python
from torchvtk.datasets import TorchDataset
cq500ds = TorchDataset.CQ500(num_workers=8)
```
This will automatically download the dataset to `~/.torchvtk/CQ500`. You can also specify a different location for your torchvtk home folder:
`TorchDataset.CQ500('/my/path/')`

If you already have the dataset extracted on your disk, you can use the script `torchvtk.converters.dicom.cq500` as follows:
```sh
python -m torchvtk.converters.dicom.cq500 /path/to/cq500 /path/to/converted/version
```
or just use our utility function from within Python:
```python
from torchvtk.converters.dicom.cq500 import cq500_to_torch
cq500_to_torch('/path/to/cq500', '/path/to/converted/version')
```

#### Convert TorchDataset to HDF5
You can convert your existing TorchDataset to a HDF5 dataset using the following script:
```
python -m torchvtk.converters.torch_to_hdf5 /path/to/TorchDataset /path/to/file.h5 -compression [lzf/gzip]
```
By specifying the `-compression` parameter you can add compression (`lzf` or `gzip`).
