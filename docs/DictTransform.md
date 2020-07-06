## DictTransform

This is the SuperClass for all of the Transformations.

```python
    tfms = NoiseDictTransform(noise_variance=(0.01, 0.1), device="cpu", apply_on=["vol"])
    tmp = tfms(file)
```




```eval_rst
.. automodule:: torchvtk.augmentation
.. autoclass:: DictTransform
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
