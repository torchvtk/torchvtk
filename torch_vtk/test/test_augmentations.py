import os

import torch
from batchviewer import view_batch

from torch_vtk.augmentation.DictTransform import BlurDictTransform, NoiseDictTransform, DictTransform, RotateDictTransform

# import test image
file_path = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Numpy", "torch", "0.pt")
file = torch.load(file_path)

# Test Noise Transform.
tfms = NoiseDictTransform(device="cpu", apply_on=["vol"], noise_variance=(0.01, 0.02))
noise_cpu = tfms(file)
del tfms
tfms = NoiseDictTransform(device="cuda", apply_on=["vol"], noise_variance=(0.01, 0.02))
noise_gpu = tfms(file)


# check for random rotation
# fixme rotation is on the wrong axis and resampling does not seem to work
tfms = RotateDictTransform(device="cpu", degree=4, axis=0, apply_on=["vol"], fillcolor_vol=0, fillcolor_mask=0)
noise_cpu = tfms(file)
noise_cpu["vol"] = noise_cpu["vol"].squeeze(0).squeeze(0)
view_batch(noise_cpu["vol"], width=512, height=512)
del tfms

# test for gaussian blur
tfms = DictTransform(apply_on=["vol"], device="cpu", channels=1, kernel_size=(3, 3, 3), sigma=1)
blur_cpu = tfms()
tmp = blur_cpu["vol"]
tmp = tmp.squeeze(0).squeeze(0)

view_batch(tmp, width=512, height=512)
tfms = DictTransform(apply_on=["vol"], device="cuda", channels=1, kernel_size=(3, 3, 3), sigma=1)
blur_gpu = tfms(file)
view_batch(blur_gpu["vol"].squeeze(0).squeeze(0), width=512, height=512)

# check for specific rotation
# todo add tests for torch 16
