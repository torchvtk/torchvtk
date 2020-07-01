## TorchQueueDataset

This Dataset holds a queue of items in memory. Being an iterable-style dataset set, it samples batches from the available queue upon demand. The queue is filled using background threads and has different filling modes.
Depending on how fast you can actually load your data compared to running your network, you might want to advance the queue by one item upon sampling (if your SSD/hard drives are fast enough). In this case use `mode="onsample"`. If you find that data loading is your bottleneck, try to make the queue as big as possible and use `mode="always"`. This will just keep pushing new items to your queue as fast as possible, removing old ones. If your network is generally faster, this is the desired way to get the most uniform sampling frequencies for all your items.


```eval_rst
.. automodule:: torchvtk.datasets
.. autoclass:: TorchQueueDataset
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```
