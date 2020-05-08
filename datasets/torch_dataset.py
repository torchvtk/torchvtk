#%%
import torch
from torch.utils.data import Dataset

from pathlib import Path
import logging
log = logging.getLogger(__name__)

from ..utils.volume_utils import make_4D


# %%
class TorchDataset(Dataset):
    def __init__(self, path, filter_fn=None, dict_keys=[]):
        super().__init__()
        self.path = Path(path)
        self.dict_keys = dict_keys
        items = self.path.rglob('*.pt')
        if filter_fn is not None:
            items = filter(filter_fn, self.items)
        self.items = list(items)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        data = torch.load(self.items[i])
        if len(self.dict_keys) > 0:
            return [make_4D(data[key]) for key in self.dict_keys]
        else:
            return make_4D(data)
