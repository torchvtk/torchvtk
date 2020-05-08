"""This script loads the numpy arrays and saves them in HDF files!"""
import os

import h5py
import numpy as np


def min_max_normalization(data, eps):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range

    return data_normalized


def main():
    # Root folder of the Training data.
    root_folder = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Numpy", "images")
    root_folder_gt = os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "Numpy", "groundtruth")

    # Hyperparameters for Training.
    seed = 77
    train_size = 0.75

    # load all arrays from the folder
    folders = os.listdir(root_folder)

    # split after seed
    split = int(train_size * len(folders))
    np.random.seed(seed)
    np.random.shuffle(folders)
    test_files, train_files = folders[split:], folders[:split]

    # Groups of the training HDF5.
    # Create the File were the Data will be stored.
    file = h5py.File(os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "train-segmentation.h5"), 'w')
    train_image_file = file.create_group("images")
    train_mask_file = file.create_group("groundtruth")

    # create train data.
    i = 0
    for x in train_files:
        train_slides = np.load(os.path.join(root_folder, x))

        # load the masks
        mask = np.load(os.path.join(root_folder_gt, x.split(".")[0] + "-gt.npy"))

        # normalization yes or no?
        train_slides = train_slides.astype(np.float32)
        train_slides = min_max_normalization(train_slides, 0.001)

        train_image_file.create_dataset(str(i), data=train_slides)
        train_mask_file.create_dataset(str(i), data=mask)
        i += 1

    file.close()
    del file, train_slides, train_image_file, train_mask_file, mask
    # Groups of the HDF5.
    file = h5py.File(os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "test-segmentation.h5"), 'w')

    test_image_file = file.create_group("images")
    test_mask_file = file.create_group("groundtruth")

    # create test data.
    i = 0
    for x in test_files:
        test_slides = np.load(os.path.join(root_folder, x))
        # load the masks
        mask = np.load(os.path.join(root_folder_gt, x.split(".")[0] + "-gt.npy"))

        # normalization yes or no?
        test_slides = test_slides.astype(np.float32)
        test_slides = min_max_normalization(test_slides, 0.001)
        # save the train slides and the masks.
        test_image_file.create_dataset(str(i), data=test_slides)
        test_mask_file.create_dataset(str(i), data=mask)
        i += 1
    file.close()


if __name__ == '__main__':
    main()
