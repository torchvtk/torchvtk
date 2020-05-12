# torch-volume-dataloaders
PyTorch volume data loading framework

Requires Python 3.6
```
cd torch-volume-dataloaders
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install gdcm -c conda-forge
pip install pydicom dicom_numpy hd5py 
pip install torch_volume_dataloaders/ext/torchsearchsorted torch_volume_dataloaders/ext/torchinterp1d
pip install -e .
```
