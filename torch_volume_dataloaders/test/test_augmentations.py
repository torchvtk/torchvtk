import os
import torch
from batchviewer import view_batch
import numpy as np
import matplotlib.pyplot as plt
import time

from torch_volume_dataloaders.augmentation.DictTransform import RotateDictTransform, BlurDictTransform, NoiseDictTransform


# import test image
file_path = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Numpy", "torch", "0.pt")

file = torch.load(file_path)

vol = file["vol"].numpy()
mask = file["mask"].numpy()

vol = vol.astype(np.float32)
print(vol.max())
print(vol.min())
print(vol.shape)


# check for random rotation
# view_batch(vol, width=512, height=512)
# fixme rotation is on the wrong axis and resampling does not seem to work
# tfms = RotateDictTransform(device="cpu", apply_on=["vol"])
# noise_cpu = tfms(file)
# noise_cpu["vol"] = noise_cpu["vol"].squeeze(0).squeeze(0)
# view_batch(noise_cpu["vol"], width=512, height=512)
# del tfms
# tfms = NoiseDictTransform(device="cuda")
#
# noise_gpu = tfms(file)
# view_batch(noise_gpu["vol"], width=512, height=512)


# test for gaussian noise
view_batch(vol.squeeze(0).squeeze(0), width=512, height=512)
tfms = NoiseDictTransform(device="cpu")
noise_cpu = tfms(file)
view_batch(noise_cpu["vol"].squeeze(0).squeeze(0), width=512, height=512)
del tfms
tfms = NoiseDictTransform(device="cuda")

noise_gpu = tfms(file)
view_batch(noise_gpu["vol"].squeeze(0).squeeze(0), width=512, height=512)
# view_batch(vol.squeeze(0).squeeze(0), width=512, height=512)


# test for gaussian blur
tfms = BlurDictTransform(1, (3,3,3))
blur_cpu = tfms(file)
tmp= blur_cpu["vol"]
tmp = tmp.squeeze(0).squeeze(0)

view_batch(tmp, width=512, height=512)
tfms = BlurDictTransform(1, (3,3,3), device="cuda")
blur_gpu = tfms(file)
view_batch(blur_gpu["vol"].squeeze(0).squeeze(0), width=512, height=512)



# check for specific rotation






