import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from ..datasets.hdf5_dataset import H5Dataset


if __name__ == '__main__':

    # Test for GPU
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if train_on_gpu else "cpu")
    pin_memory = True

    batch_size = 2
    num_workers = 4
    epoch = 10
    dataset = H5Dataset(
            h5_dir=os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "train-segmentation.h5"))

    num_train = len(dataset)
    print("Amount of Samples: ", num_train)
    indices = list(range(num_train))

    # Generating train and validation dataloader.
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    for i in range(epoch):

        # Reset Metrics
        since = time.time()
        for x, y in train_loader:
            # Put the Images & Groundtruth on the GPU.

            x, y = x.to(device).unsqueeze(1).float(), y.to(device).to(torch.long)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

