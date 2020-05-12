#%%
#%%% load_ext autoreload 2
import torch
import numpy as np
from torch_volume_dataloaders.preprocessing.dicom.utils import *
from torch_volume_dataloaders.preprocessing.dicom.cq500 import get_volume_gen, traverse_cq500_folders


# %%
qure_path = 'C:/Users/domin/Dev/torch-volume-dataloaders/torch_volume_dataloaders/data/Qure_AI_Brain_CT'
vol_dirs = traverse_cq500_folders(qure_path, 100, 700)
# ds = get_volume_gen()

# %%
vol_gen = get_volume_gen(vol_dirs)

# %%
for vol, vox_scl, vol_name in vol_gen:
    if   vol.flags['C_CONTIGUOUS']: contig = 'C'
    elif vol.flags['F_CONTIGUOUS']: contig = 'F'
    else:                           contig = "not"
    print(f'Volume {vol_name} with shape {vol.shape} (Vox Scale: {vox_scl}) is {contig} contiguous.')

# %%
