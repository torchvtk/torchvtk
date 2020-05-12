import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from torch_volume_dataloaders.datasets.hdf5_dataset import H5Dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset = H5Dataset(
            h5_dir=os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "speedtest_hdf5-gzip.h5"))

    amount_of_examples = 50

    times = []
    since = time.time()
    for i in range(amount_of_examples):
        since1 = time.time()
        x, y = dataset[i]
        time_elapsed1 = time.time() - since1
        print('{:.0f}m {:.0f}s'.format(time_elapsed1 // 60, time_elapsed1 % 60))
        times.append(time_elapsed1)

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    # create histogram
    plt.hist(times, bins=5, color='c')
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.savefig("hdf5.png")