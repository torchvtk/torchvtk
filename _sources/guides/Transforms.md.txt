## Transforms
### Usage
Transforms include common data augmentation and preprocessing functions that you would use with volumetric data in a machine learning context. We implement all transforms directly in PyTorch, therefore they should all work on both CPU and GPU and do not require further dependencies. If you would like to see transforms that we have not implemented, feel free to write an issue or send us a PR.

For Transforms in `torchvtk` it is necessary to inherit `DictTransform`.
A `DictTransform` takes care of applying transformations to specified items of the data dict (as defined in [TorchDataset](TorchDataset.html)) and gives control over `dtype` and `device` for the transform. We stick with this dictionariy paradigm to chain all possible preprocessing tasks together in a `Composite`.
An example is presented in the following code snipped.

```python
    from torchvtk.transforms import Composite, RandFlip, Resize, Lambda
    def _basic_func(data):
        data['newkey'] = 1 # do some custom stuff here
        return data        # Make sure to return the dictionary!

    def _basic_add(number): # This function does not care about dict's
       return number + 1

    tfms = Composite(
      _basic_func,
      Lambda(_basic_add, apply_on='newkey'),
      RandFlip(),
      Resize((256,256,256),
      apply_on=['vol'])
    tmp = tfms(data) # data should be a dict, like the item of a TorchDataset

    ds = TorchDataset(path_to_ds, preprocess_fn=tfms) # Use the transforms upon loading with a TorchDataset
```
#### Compositing
We can Composite `DictTransform`s, as well as normal functions, assuming they all operate on `dict`s and return the modified dictionary. All subclasses of `DictTransform`, which are all classes in `torchvtk.transforms`, can be given the parameters of `DictTransform` through the `**kwargs` in their respective `__init__`s. That means you can specify for each transform on which items in the dictionary they should be applied (e.g. `apply_on=['vol']`), as well as a preferred `torch.dtype` and `torch.device` for the transform.

Note that setting the `apply_on` paramter of `Composite` (as in the example), applies to all transforms that have not specified `apply_on` themselves. The `dtype` and `device` parameters work similar.

#### DictTransform arguments
Further note that by default the transformations are applied to all keys that have a `torch.Tensor` as value. Beware of other Tensors in your data that you do not wan't to modify! You should usually define `apply_on` for all your transforms somehow, be it specific or through `Composite`. As for the `dtype` and `device`, the transform is executed on using the type and device that the data comes in. Setting a type or device at the beginning of a `Composite` can thus determine the type or device until the next transform specifies another.

#### Mixing with non-DictTransforms
In this example, the `_basic_func` is executed first and simply gets the whole dictionary in and must return the modified one. Here a new key `'newkey'` is added. `_basic_add` is a standard function that knows nothing of dicts and we can wrap it using `Lambda` to make use of `apply_on` etc.
As you can see we apply `_basic_add` only to `'newkey'`. All the other transforms in the Composite are applied to `'vol'`, since the `Composite` sets it for all transforms that did not specify `apply_on`.

### API
#### DictTransform
```eval_rst
.. autoclass:: torchvtk.transforms::DictTransform
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
```

#### Composite
```eval_rst
.. autoclass:: torchvtk.transforms::Composite
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### Lambda
```eval_rst
.. autoclass:: torchvtk.transforms::Lambda
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### Crop
```eval_rst
.. autoclass:: torchvtk.transforms::Crop
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### Resize
```eval_rst
.. autoclass:: torchvtk.transforms::Resize
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### RandFlip
```eval_rst
.. autoclass:: torchvtk.transforms::RandFlip
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### RandPermute
```eval_rst
.. autoclass:: torchvtk.transforms::RandPermute
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### GaussianBlur
```eval_rst
.. autoclass:: torchvtk.transforms::GaussianBlur
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

#### GaussianNoise
```eval_rst
.. autoclass:: torchvtk.transforms::GaussianNoise
   :noindex:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```
