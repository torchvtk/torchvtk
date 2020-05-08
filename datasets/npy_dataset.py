#%%
import torch
from torch.utils.data import Dataset
import numpy as np

from pathlib import Path
import logging
log = logging.getLogger(__name__)

#%%
class NumpyDataset(Dataset):
    def __init__(self, path, filter_fn=None):
        super().__init__()
        self.path = Path(path)
        items = self.path.rglob('*.npy')
        if filter_fn is not None:
            items = filter(filter_fn, self.items)
        self.items = list(items)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        vol = np.load(self.items[i])
        if   vol.ndim == 4: return torch.from_numpy(vol)
        elif vol.ndim == 3: return torch.from_numpy(vol.unsqueeze(0))
        else:
            error_string = f'Loaded volume ({self.items[i]}) has shape {vol.shape}. Expected number of dimensions to be 3 or 4 (Found {vol.ndim})'
            log.error(error_string)
            raise Exception(error_string)
