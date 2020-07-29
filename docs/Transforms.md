## Transformations

For Transformations it is necessary to call the Subclasses of `DictTransform`.
An example is presented in the following code snipped.

```python
    from torchvtk.transforms import Composite, RandFlip, Resize
    def _basic_func(data):
        data['newkey'] = 1 # do some custom stuff here
        return data        # Make sure to return the dictionary!

    tfms = Composite(
      RandFlip(),
      Resize((256,256,256),
      apply_all_on=['vol'])
    tmp = tfms(data) # data should be a dict, like the item of a TorchDataset

    ds = TorchDataset(path_to_ds, preprocess_fn=tfms) # Use the transforms upon loading with a TorchDataset

```
We can Composite `DictTransform`s, as well as normal functions, assuming they all operate on `dict`s and return the modified dictionary. All subclasses of `DictTransform`, which are all classes in `torchvtk.transforms` except for the `Composite`, can be given the parameters of `DictTransform` through the `**kwargs` in their respective `__init__`s. That means you can specify for each transform on which items in the dictionary they should be applied (e.g. `apply_on=['vol']`), as well as a preferred `torch.dtype` and `torch.device` for the transform.

Note that setting the `apply_all_on` paramter of `Composite` (as in the example), overrides all specifically set transforms' `apply_on`s.


```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: DictTransform
   :members:
   :undoc-members:
   :inherited-members:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: Composite
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: Lambda
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: Crop
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: Resize
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: RandFlip
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: RandPermute
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: GaussianBlur
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

```eval_rst
.. automodule:: torchvtk.transforms
.. autoclass:: GaussianNoise
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
