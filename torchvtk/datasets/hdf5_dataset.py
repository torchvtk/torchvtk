import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class H5Dataset(Dataset):
    """
    Loading the HDF5 Data Sets.
    """

    def __init__(self, h5_dir):
        self.root_dir = h5_dir
        self.dataset = None
        with h5py.File(self.root_dir, 'r') as file:
            self.dataset_len = len(file["images"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.dataset is None:
            image = h5py.File(self.root_dir, 'r')["images"][str(idx)]
            image = np.array(image)
            mask = h5py.File(self.root_dir, 'r')["groundtruth"][str(idx)]
            mask = np.array(mask)

        # kann auch als numpy array geladen werden, eventuell noch permutation, data type changes.
        return torch.from_numpy(image), torch.from_numpy(mask)

def to_torch_if_numpy(e):
    return torch.tensor(e) if len(e.shape) > 0 else e

class H5DatasetReopen(Dataset):
    def __init__(self, h5_path, preprocess_fn=None):
        super().__init__()
        self.root_dir = h5_path
        self.preprocess_fn = preprocess_fn
        with h5py.File(self.root_dir, 'r') as f:
            self.ds_len = len(list(f.values())[0])

    def __len__(self): return self.ds_len

    def __getitem__(self, i):
        with h5py.File(self.root_dir, 'r') as f:
            data = {name: to_torch_if_numpy(group[str(i)]) for name, group in f.items()}
            if self.preprocess_fn is not None:
                  return self.preprocess_fn(data)
            else: return data

class H5DatasetOpenOnce(Dataset):
    def __init__(self, h5_path, preprocess_fn=None):
        super().__init__()
        self.root_dir = h5_path
        self.preprocess_fn = preprocess_fn
        self.f = h5py.File(self.root_dir, 'r')
        self.ds_len = len(list(self.f.values())[0])

    def __len__(self): return self.ds_len

    def __getitem__(self, i):
        data = {name: to_torch_if_numpy(group[str(i)]) for name, group in self.f.items()}
        if self.preprocess_fn is not None:
              return self.preprocess_fn(data)
        else: return data
