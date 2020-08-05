## Getting Started

This document should get you up and running with `torchvtk`!

### Installation
#### Latest GitHub release
```
pip install torchvtk
```
#### Latest master release
```
pip install git+https://github.com/xeTaiz/torchvtk.git@master#egg=torchvtk
```
#### Developer Installation
This will directly include your local changes to `torchvtk`.
```
git clone https://github.com/xeTaiz/torchvtk.git
cd torchvtk
pip install -e .
```


### Features
#### Fast binary-format Datasets
[We have benchmarked](Benchmarks.html) different ways to load large binary data for use with PyTorch. We have defined a standard representation for volume data for our framework, which is based on saving individual volumes with all their metadata and additional info as a dict, serialized using PyTorch's pickler. As a result you can easily load single items using `torch.load` and you can define your own datasets with your special needs, without having to fight against our framework. Just add whatever you need to the dictionary, the rest of the framework is based on this structure

#### Cached Queue
Volume data is always large and the data loading can significantly slow down your training. The [TorchQueueDataset](TorchQueueDataset.html) allows you to cache the data in system (or GPU) memory easily to decouple your training speeds for the slow data loading. The queue loads data as fast as you train, or as fast as you can load from disk, while allowing your network to be trained off the RAM cache.

#### Easy Preprocessing
Learn about `torchvtk.transforms` [here](Transforms.html). These are some widely used transformations for volumes, all implemented in PyTorch. That means you can run them on the GPU as well, if you want.
But besides the runtime transforms, we also made it easy to create new preprocessed datasets on disk, to reduce the overhead during runtime. If your preprocessing includes downsampling of the volume, this can also drastically reduce your dataset size and thus affest loading speeds. Check out [our guide](TorchDataset.html#more-involved-example)

#### Automatic Dataset Downloading
We have a couple of datasets that can be directly downloaded, unpacked and converted to our format available through `TorchDataset`'s statics.

#### Volume Rendering Tools
`torchvtk.rendering` includes the `VolumeRaycaster`, a differentiable volume raycaster.

#### Volume Handling Tools
In `torchvtk.utils` you can find a variety of tools regarding volumes. There are some functions to handle volume tensors in general, deal with scales, etc.
Furthermore there are transfer function utils that can convert between point-based TFs and texture-based TFs, as well as generate random TFs from volume histograms.

### Contributing
At the moment the project is rather small and we are open for all suggestions right on our [GitHub](https://github.com/xeTaiz/torchvtk). Just throw us an issue ;)
