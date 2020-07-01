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
    train_size = 1

    # load all arrays from the folder
    folders = os.listdir(root_folder)

    # split after seed
    split = int(train_size * len(folders))
    np.random.seed(seed)
    np.random.shuffle(folders)
    test_files, train_files = folders[split:], folders[:split]

    # Groups of the training HDF5.
    # Create the File were the Data will be stored.
    file = h5py.File(os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "speedtest_hdf5.h5"), 'w')
    train_image_file = file.create_group("images")
    train_mask_file = file.create_group("groundtruth")

    # create train data.
    i = 0
    for x in train_files:
        train_slides = np.load(os.path.join(root_folder, x))

        # load the masks
        mask = np.load(os.path.join(root_folder_gt, x.split(".")[0] + "-gt.npy"))

        # normalization yes or no?
        # hyperparameter maybe float16 for amp.
        train_slides = train_slides.astype(np.float32)
        train_slides = min_max_normalization(train_slides, 0.001)

        train_image_file.create_dataset(str(i), data=train_slides)
        train_mask_file.create_dataset(str(i), data=mask)
        i += 1

    file.close()
    print("Amount of images", i-1)
    del file, train_slides, train_image_file, train_mask_file, mask



    # COMPRESSION GZIP
    file = h5py.File(os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "speedtest_hdf5-gzip.h5"), 'w')
    train_image_file = file.create_group("images")
    train_mask_file = file.create_group("groundtruth")

    # create train data.
    i = 0
    for x in train_files:
        train_slides = np.load(os.path.join(root_folder, x))

        # load the masks
        mask = np.load(os.path.join(root_folder_gt, x.split(".")[0] + "-gt.npy"))

        # normalization yes or no?
        # hyperparameter maybe float16 for amp.
        train_slides = train_slides.astype(np.float32)
        train_slides = min_max_normalization(train_slides, 0.001)

        train_image_file.create_dataset(str(i), compression="gzip", compression_opts=9, data=train_slides)
        train_mask_file.create_dataset(str(i), compression="gzip", compression_opts=9, data=mask)
        i += 1

    file.close()
    print("Amount of images", i - 1)
    del file, train_slides, train_image_file, train_mask_file, mask



    # COMPRESSION LZF
    file = h5py.File(os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "speedtest_hdf5-lzf.h5"), 'w')
    train_image_file = file.create_group("images")
    train_mask_file = file.create_group("groundtruth")

    # create train data.
    i = 0
    for x in train_files:
        train_slides = np.load(os.path.join(root_folder, x))

        # load the masks
        mask = np.load(os.path.join(root_folder_gt, x.split(".")[0] + "-gt.npy"))

        # normalization yes or no?
        # hyperparameter maybe float16 for amp.
        train_slides = train_slides.astype(np.float32)
        train_slides = min_max_normalization(train_slides, 0.001)

        train_image_file.create_dataset(str(i), compression="lzf", data=train_slides)
        train_mask_file.create_dataset(str(i), compression="lzf", data=mask)
        i += 1

    file.close()
    print("Amount of images", i - 1)
    del file, train_slides, train_image_file, train_mask_file, mask



    # COMPRESSION SZIP
    file = h5py.File(os.path.join("D:", os.sep, "DownloadDatasets", "medical_decathlon", "speedtest_hdf5-szip.h5"),
                     'w')
    train_image_file = file.create_group("images")
    train_mask_file = file.create_group("groundtruth")

    # create train data.
    i = 0
    for x in train_files:
        train_slides = np.load(os.path.join(root_folder, x))

        # load the masks
        mask = np.load(os.path.join(root_folder_gt, x.split(".")[0] + "-gt.npy"))

        # normalization yes or no?
        # hyperparameter maybe float16 for amp.
        train_slides = train_slides.astype(np.float32)
        train_slides = min_max_normalization(train_slides, 0.001)

        train_image_file.create_dataset(str(i), compression="szip", data=train_slides)
        train_mask_file.create_dataset(str(i), compression="szip", data=mask)
        i += 1

    file.close()
    print("Amount of images", i - 1)
    del file, train_slides, train_image_file, train_mask_file, mask


if __name__ == '__main__':
    main()
