## Transformations

For Transformations it is necessary to call the Subclasses of the DictTransform Super Class.
An example is presented in the following code snipped.

```python
    tfms = NoiseDictTransform(noise_variance=(0.01, 0.1), device="cpu", apply_on=["vol"])
    tmp = tfms(file)
```




```eval_rst
.. automodule:: torchvtk.augmentation
.. autoclass:: DictTransform 
   :members: NoiseDictTransform
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
