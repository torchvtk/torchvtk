import os

import torch
from batchviewer import view_batch

from torch_volume_dataloaders.augmentation.DictTransform import BlurDictTransform, NoiseDictTransform, DictTransform, RotateDictTransform

# import test image
file_path = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Numpy", "torch", "0.pt")

file = torch.load(file_path)

# Test Dict Transform.

tfms = DictTransform(func=NoiseDictTransform, device="cpu", apply_on=["vol"])

noise_cpu = tfms(file, noise_variance=(0.01, 0.02))
del tfms
tfms = DictTransform(func=NoiseDictTransform, device="cuda", apply_on=["vol"])
noise_gpu = tfms(file, noise_variance=(0.01, 0.02))
#
# check for random rotation
# fixme rotation is on the wrong axis and resampling does not seem to work
tfms = DictTransform(func=RotateDictTransform, device="cpu")
noise_cpu = tfms(file, degree=4, axis=0, apply_on=["vol"], fillcolor_vol=0, fillcolor_mask=0)
noise_cpu["vol"] = noise_cpu["vol"].squeeze(0).squeeze(0)
view_batch(noise_cpu["vol"], width=512, height=512)
del tfms
#
# noise_gpu = tfms(file)
# view_batch(noise_gpu["vol"], width=512, height=512)


# test for gaussian noise
# view_batch(vol.squeeze(0).squeeze(0), width=512, height=512)
# tfms = NoiseDictTransform(device="cpu")
# noise_cpu = tfms(file)
# view_batch(noise_cpu["vol"].squeeze(0).squeeze(0), width=512, height=512)
# del tfms
# tfms = NoiseDictTransform(device="cuda")
#
# noise_gpu = tfms(file)
# view_batch(noise_gpu["vol"].squeeze(0).squeeze(0), width=512, height=512)
# view_batch(vol.squeeze(0).squeeze(0), width=512, height=512)


# test for gaussian blur
tfms = DictTransform(func=BlurDictTransform, apply_on=["vol"])
blur_cpu = tfms(file, channels=1, kernel_size=(3, 3, 3), sigma=1)
tmp = blur_cpu["vol"]
tmp = tmp.squeeze(0).squeeze(0)

view_batch(tmp, width=512, height=512)
tfms = DictTransform(func=BlurDictTransform, apply_on=["vol"], device="cuda")
blur_gpu = tfms(file, channels=1, kernel_size=(3, 3, 3), sigma=1)
view_batch(blur_gpu["vol"].squeeze(0).squeeze(0), width=512, height=512)

# check for specific rotation
# todo add tests for torch 16
