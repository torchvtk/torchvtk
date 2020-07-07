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
   :members: 
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
```eval_rst
.. automodule:: torchvtk.augmentation
.. autoclass:: NoiseDictTransform 
   :members: 
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.augmentation
.. autoclass:: CroppingTransform 
   :members: 
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.augmentation
.. autoclass:: BlurDictTransform 
   :members: 
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
