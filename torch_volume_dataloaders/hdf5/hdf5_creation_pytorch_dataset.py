import os

import h5py
import torch

class HDF5_Pytorch():

    def __init__(self):
        self.h5file = h5py.File(
            os.path.join("blubb.h5"), 'w')
        self.images = self.h5file.create_group("images")
        self.groundtruth = self.h5file.create_group("groundtruth")
        self.i = 0

    def __iter__(self, dataset):
        for x, y in dataset:
            x = x.numpy()
            y = y.numpy()

            self.images.create_dataset(str(self.i), data=x)
            self.groundtruth.create_dataset(str(self.i), data=y)

            self.i += 1

    def __close__(self):
        self.h5file.close()
