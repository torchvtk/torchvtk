#%%
import torch
from torch.utils.data import Dataset

from pathlib import Path


class TorchDataset(Dataset):
    ''' A Dataset loading serialized PyTorch tensors from disk. '''
    def __init__(self, ds_files, filter_fn=None, preprocess_fn=None):
        ''' Initializes TorchDataset
        Args:
            ds_files (str, Path (Dict), List of Path (Files)): Path to the TorchDataset directory (containing *.pt) or list of paths pointing to .pt files
            filter_fn (function): Function that filters the found items. Input is filepath
            preprocess_fn (function): Function to process the loaded dirctionary. '''
        super().__init__()
        self.preprocess_fn = preprocess_fn
        if  isinstance(ds_files, (str, Path)):
            self.path = Path(ds_files)
            assert self.path.is_dir()
            items = self.path.rglob('*.pt')
            if filter_fn is not None:
                items = filter(filter_fn, items)
            self.items = list(items)
        elif isinstance(ds_files, (list, tuple)):
            for f in ds_files:
                assert Path(f).is_file() and Path(f).suffix == '.pt'
            self.path = ds_files[0].parent
            if filter_fn is not None:
                  self.items = list(filter(filter_fn, ds_files))
            else: self.items = list(ds_files)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        data = torch.load(self.items[i])
        if self.preprocess_fn is not None:
              return self.preprocess_fn(data)
        else: return data
