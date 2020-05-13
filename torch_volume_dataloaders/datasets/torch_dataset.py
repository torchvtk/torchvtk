#%%
import torch
from torch.utils.data import Dataset

from pathlib import Path

# %%
class TorchDataset(Dataset):
    def __init__(self, path, filter_fn=None, preprocess_fn=None):
        super().__init__()
        self.path = Path(path)
        self.preprocess_fn = preprocess_fn
        items = self.path.rglob('*.pt')
        if filter_fn is not None:
            items = filter(filter_fn, self.items)
        self.items = list(items)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        data = torch.load(self.items[i])
        if self.preprocess_fn is not None:
              return self.preprocess_fn(data)
        else: return data
