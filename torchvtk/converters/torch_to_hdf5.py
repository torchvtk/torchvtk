from pathlib  import Path
from argparse import ArgumentParser

import torch
import h5py

from torchvtk.datasets import TorchDataset

def torch_to_hdf5(torch_ds, hdf5_path, compression=None):
    ''' Converts a given TorchDataset `torch_ds` to a HDF5 Dataset and saves it to a given path
    Args:
        torch_ds (TorchDataset): A TorchDataset that retuns  dictionaries. All items in the dict are serialized as separate groups.
        hdf5_path (string, Path): Path to save the HDF5 file
    '''
    if isinstance(compression, str): compression = compression.lower()
    assert compression in [None, 'gzip', 'lzf']
    with h5py.File(hdf5_path, 'w') as h5file:
        sample = torch_ds[0] # Get first item to see what data is available
        groups = {g: h5file.create_group(g) for g in sample.keys()} # Create group for each key in sample dict
        for i, it in enumerate(torch_ds):
            for k, v in it.items():
                data = v.numpy() if torch.is_tensor(v) else v # Convert tensors to numpy arrays
                groups[k].create_dataset(str(i), data=data, compression=compression if torch.is_tensor(v) else None)

if __name__ == '__main__':
    parser = ArgumentParser("TorchDataset to HDF5Dataset Converter", description='''
        Reads a TorchDataset from disk, converts it to HDF5 and saves it to a given destination.
    ''')
    parser.add_argument('torch_path',  type=str, help='Path to the TorchDataset')
    parser.add_argument('hdf5_path',   type=str, help='Path to the HDF5 Dataset destination.')
    parser.add_argument('-compression',type=str, default=None, help='Compression to use in H5Dataset. Must be either gzip or lzf')
    args = parser.parse_args()

    torch_ds = TorchDataset(args.torch_path)
    torch_to_hdf5(torch_ds, args.hdf5_path, compression=args.compression)
