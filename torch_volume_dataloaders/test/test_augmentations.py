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


view_batch(vol, width=512, height=512)

tfms = NoiseDictTransform(device="cpu")
noise_cpu = tfms(file)
view_batch(noise_cpu["vol"], width=512, height=512)
del tfms
tfms = NoiseDictTransform(device="cuda")

noise_gpu = tfms(file)
view_batch(noise_gpu["vol"], width=512, height=512)





# check for random rotation


# check for specific rotation


# test for gaussian noise


# test for gaussian blur


