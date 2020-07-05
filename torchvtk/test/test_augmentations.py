import os

import torch
from batchviewer import view_batch

from torchvtk.augmentation.DictTransform import BlurDictTransform, NoiseDictTransform, CroppingTransform

# import test image
file_path = "data/test_ct_images.pt"
file = torch.load(file_path)


file["vol"] = torch.stack([file["vol"], file["vol"]], dim=0)
view_batch(file["vol"][0, ...], width=512, height=512)
# Test Noise Transform.
tfms = NoiseDictTransform(device="cpu", apply_on=["vol"], noise_variance=(0.01, 0.1), batch_transform=True)
noise_cpu = tfms(file).copy()
view_batch(noise_cpu["vol"][0, ...], width=512, height=512)
view_batch(noise_cpu["vol"][1, ...], width=512, height=512)
del tfms
tfms = NoiseDictTransform(device="cuda", apply_on=["vol"], noise_variance=(0.01, 0.02))
noise_gpu = tfms(file)

file["vol"] = file["vol"].to("cpu")
# test for gaussian blur
tfms = BlurDictTransform(apply_on=["vol"], device="cpu", channels=1, kernel_size=(3, 3, 3), sigma=1)
blur_cpu = tfms(file)
tmp = blur_cpu["vol"]
tmp = tmp.squeeze(0).squeeze(0)

view_batch(tmp, width=512, height=512)
tfms = BlurDictTransform(apply_on=["vol"], device="cuda", channels=1, kernel_size=(3, 3, 3), sigma=1)
blur_gpu = tfms(file)
view_batch(blur_gpu["vol"].squeeze(0).squeeze(0), width=512, height=512)
file["vol"] = file["vol"].to("cpu")

# Cropping
view_batch(file["vol"], width=512, height=512)
tfms = CroppingTransform(device="cuda",  apply_on=["vol"], dtype=torch.float32)
noise_cpu = tfms(file)
noise_cpu["vol"] = noise_cpu["vol"].squeeze(0).squeeze(0)
view_batch(noise_cpu["vol"], width=512, height=512)
del tfms
