"""This script works for the medical decathlon files."""

import os

import nibabel as nib
import numpy as np

# source path of image and groundtruth segmentation.
images_path = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Task06_Lung", "imagesTr")
label_path = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Task06_Lung", "labelsTr")

# Path where the numpy arrays will be saved.
numpy_path = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Numpy")

# crawl the training images
for files in os.listdir(images_path):
    # combine the nifti files to numpy array.
    img = nib.load(os.path.join(images_path, files))
    img = img.get_fdata()
    img = img.transpose(2, 0, 1)
    print(img.shape)
    # creates channels first
    np.save(os.path.join(numpy_path, "images", str(files).split(".")[0] + ".npy"), img)

# crawl the training groundtruth
for files in os.listdir(label_path):
    img = nib.load(os.path.join(label_path, files))
    img = img.get_fdata()
    img = img.transpose(2, 0, 1)
    # creates channels first
    np.save(os.path.join(numpy_path, "groundtruth", str(files).split(".")[0] + "-gt.npy"), img)
