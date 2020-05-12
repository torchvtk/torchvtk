# torch-volume-dataloaders
PyTorch volume data loading framework

Requires Python 3.6
```
cd torch-volume-dataloaders
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install gdcm -c conda-forge
pip install pydicom dicom_numpy h5py 
pip install torch_volume_dataloaders/ext/torchsearchsorted torch_volume_dataloaders/ext/torchinterp1d
pip install -e .
```

### Creating the Numpy Files for the HDF 5 File generation.

`hdf5/nifiti_crawler.py` This script generates out of the nifti Files of the Medical Decathlon Challenge numpy arrays that contain the images and the segmentation groundtruths.



### Creating the HDF5 Files

`hdf5/hdf5_crawler.py` This script generates hdf5 Files with different compression techniques. These files contain the image as well as the groundtruth. The image is normalized between [0,1] and is stored in the Float32 Format. The Groundtruth is saved as an Int16 Format. The used compressions are gzip, szip  and lzf.
