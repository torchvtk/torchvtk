import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from torchvtk.datasets.hdf5_dataset import H5Dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset = H5Dataset(
            h5_dir=os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "speedtest_hdf5.h5"))

    amount_of_examples = 5

    times = []
    for i in range(amount_of_examples):
        before = time.time()
        x, y = dataset[i]
        after = time.time()
        times.append(after - before)

    times = np.array(times)
    avg_time = np.mean(times)
    max_time = np.max(times)
    total_time = np.sum(times)

    print("average time per sample", avg_time)
    print("maximal time", max_time)
    print("total time", total_time)

    # create histogram
    plt.hist(times, color='b')
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.savefig("hdf5.png")
