# torchvtk
PyTorch volume data loading framework
## [Documentation](https://torchvtk.github.io)


## Installation Instructions
The latest GitHub release is pushed to PyPi:
```
pip install torchvtk
```

To get the master run:
```
pip install git+https://github.com/torchvtk/torchvtk.git@master#egg=torchvtk
```

### Optional for DICOM stuff only:
```
conda create --name "tvtk" python=3.6 && conda activate tvtk
conda install gdcm -c conda-forge
pip install pydicom dicom_numpy h5py numpy matplotlib
```
If you need DICOM, and thus gdcm, your Python version needs to be <=3.6
Modify `tvtk` in the third line (both after `--name` and at the end of the line) to your preferred environment name or just add the required packages to your existing environment.
Note that the restriction to Python <= 3.6 is due to `gdcm` and higher version should work as well if you don't need DICOM loading capabilities.

### Creating the Numpy Files for the HDF 5 File generation.
`hdf5/nifiti_crawler.py` This script generates out of the nifti Files of the Medical Decathlon Challenge numpy arrays that contain the images and the segmentation groundtruths.

### Creating the HDF5 Files
`hdf5/hdf5_crawler.py` This script generates hdf5 Files with different compression techniques. These files contain the image as well as the groundtruth. The image is normalized between [0,1] and is stored in the Float32 Format. The Groundtruth is saved as an Int16 Format. The used compressions are gzip, szip  and lzf.
