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
