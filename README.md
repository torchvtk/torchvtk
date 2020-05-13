# torch-volume-dataloaders
PyTorch volume data loading framework

### Installation Instructions
Requires Python 3 <= 3.6, we recommend using a conda environment with the following setup for development
```
git clone --recurse-submodules https://github.com/xeTaiz/torch-volume-dataloaders.git
cd torch-volume-dataloaders
conda create --name "tvdls" python=3.6 && conda activate tvdls
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install gdcm -c conda-forge
pip install pydicom dicom_numpy h5py numpy matplotlib
pip install ext/torchsearchsorted ext/torchinterp1d
pip install -e .
```
Modify `tvdls` in the third line (both after `--name` and at the end of the line) to your preferred environment name or just add the required packages to your existing environment.
Note that the restriction to Python <= 3.6 is due to `gdcm` and higher version should work as well if you don't need DICOM loading capabilities.

### Creating the Numpy Files for the HDF 5 File generation.
`hdf5/nifiti_crawler.py` This script generates out of the nifti Files of the Medical Decathlon Challenge numpy arrays that contain the images and the segmentation groundtruths.



### Creating the HDF5 Files
`hdf5/hdf5_crawler.py` This script generates hdf5 Files with different compression techniques. These files contain the image as well as the groundtruth. The image is normalized between [0,1] and is stored in the Float32 Format. The Groundtruth is saved as an Int16 Format. The used compressions are gzip, szip  and lzf.
