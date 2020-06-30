#%%
import torch

from torchvtk.datasets.queue import TorchQueueDataset
from torchvtk.datasets.torch_dataset import TorchDataset


# %%
path = '/run/media/dome/Data/data/Volumes/tvdls'
ds = TorchDataset(path)


# %%
qds = TorchQueueDataset(ds, num_workers=2, q_maxlen=10, wait_fill=4, mode='always', bs=2)
qds.qsize

# %%
import psutil
mem = psutil.virtual_memory().available

# %%
