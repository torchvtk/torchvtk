#%%
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from pathlib import Path
from functools import partial
from bisect import bisect
import shutil, os, pprint
import torchvtk.datasets.urls as urls
from torchvtk.datasets.download import download_all, extract_all
from torchvtk.converters.dicom import cq500_to_torch, test_has_gdcm
from torchvtk.utils import pool_map, make_5d, clone
from torchvtk.transforms import Resize

def _preload_dict_tensors(it, device='cpu'):
    for k, v in it.items():
        if torch.is_tensor(v): it[k] = v.to(device)
    return it

class DatasetWork:
    def __init__(self, items, target_path, process_fn):
        self.items = items
        self.target_path = target_path
        self.process_fn = process_fn
    def __call__(self, i):
        fn = self.items[i]
        tfn = self.target_path/fn.name

        torch.save(self.process_fn(torch.load(fn)), tfn)

class TorchDataset(Dataset):
    def __init__(self, ds_files, filter_fn=None, preprocess_fn=None):
        ''' A dataset that uses serialized PyTorch Tensors.

        Args:
            ds_files (str, Path (Dict), List of Path (Files)): Path to the TorchDataset directory (containing `*.pt`) or list of paths pointing to .pt files
            filter_fn (function): Function that filters the found items. Input is filepath
            preprocess_fn (function): Function to process the loaded dirctionary.
        '''
        super().__init__()
        self.preprocess_fn = preprocess_fn
        self.filter_fn     = filter_fn
        if  isinstance(ds_files, (str, Path)):
            self.path = Path(ds_files)
            if self.path.is_dir():
                items = self.path.rglob('*.pt')
                if filter_fn is not None:
                    items = filter(filter_fn, items)
                self.items = list(items)
            elif self.path.is_file():
                raise Exception(f"Given ds_files is the path to a file. Use TorchDataset.from_file() instead.")
            else:
                raise Exception(f"Given ds_files is no valid path: {self.path}")
        elif isinstance(ds_files, (list, tuple)):
            assert len(ds_files) > 0
            for f in ds_files:
                assert Path(f).is_file() and Path(f).suffix == '.pt'
            self.path = ds_files[0].parent
            if filter_fn is not None:
                  self.items = list(filter(filter_fn, ds_files))
            else: self.items = list(ds_files)

    @staticmethod
    def from_file(file_path, filter_fn=None, preprocess_fn=None):
        path = Path(file_path)
        assert path.exists() and path.is_file()
        ds = TorchDataset([path], filter_fn=filter_fn, preprocess_fn=preprocess_fn)
        return PreloadedTorchDataset(ds, override_data=torch.load(str(path)))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        data = torch.load(str(self.items[i]))
        if self.preprocess_fn is not None:
              return self.preprocess_fn(data)
        else: return data

    def cache_processed(self, process_fn, name, num_workers=0, delete_old_from_disk=False):
        ''' Processes the given TorchDataset and serializes it.
        Iterates through the dataset and applies the given `process_fn` to each item (which should be a dictionary).
        The resulting new dataset will be serialized next to the old one, using then given `name`.
        This function can work multithreaded.

        Args:
            process_fn (function): The function to be applied on the inidividual items
            name (str): Name of the new processed dataset
            num_workers (int > 0): Number of threads used for processing
            delete_old_from_disk (bool): If True, the root directory of the old, unprocessed, dataset is removed from disk.

        Returns:
            TorchDataset: TorchDataset with the new items. (no filter or preprocess_fn set)
        '''
        target_path = self.path.parent/name
        target_path.mkdir()
        print(f'Preprocessing TorchDataset ({self.path}) to {target_path}...')

        work = DatasetWork(self.items, target_path, process_fn)
        pool_map(work, [i for i in range(len(self))], num_workers=num_workers)

        items = target_path.rglob('*.pt')
        assert len(list(items)) == len(self)
        if delete_old_from_disk: shutil.rmtree(self.path)
        return TorchDataset(target_path)

    def preload(self, device='cpu', num_workers=0):
        ''' Preloads the dataset into memory.

        Args:
            device (torch.device, optional): Device to store the dataset on. Defaults to 'cpu'.
            num_workers (int, optional): Number of workers to load items into memory. Defaults to 0.

        Returns:
            PreloadedTorchDataset: New TorchDataset using the preloaded data.
        '''
        return PreloadedTorchDataset(self, device=device, num_workers=num_workers)

    def tile(self, keys_to_tile, tile_sz=128, overlap=2, dim=3, **kwargs):
        ''' Converts the Dataset to a Tiled Dataset, drawing only parts of the data
        Since the data needs to be loaded to determine the number of tiles, a tile is drawn randomly after loading the volume, Without guaranteeing full coverage.

        Args:
            keys_to_tile ([str]): List of strings matching the keys of the data dictionaries that need to be tiled. All must have the same shape and result in the same tiling.
            tile_sz (int/tuple of ints, optional): Size of the tiles drawn. Either int or tuple with length matching the given `dim`. Defaults to 128.
            overlap (int/tuple of its, optional): . Defaults to 2.
            dim (int, optional): Dimensionality of the data. If `tile_sz` or `overlap` is given as tuple this must match their lengths. Defaults to 3.

        Returns:
            TiledTorchDataset: Tiling-aware `TorchDataset`
        '''
        return TiledTorchDataset(self, keys_to_tile,
            tile_sz=tile_sz,
            overlap=overlap,
            dim=dim,
            **kwargs)

    def __repr__(self):
        it = self[0]
        if isinstance(it, dict):
            def _format_object(t):
                if   torch.is_tensor(t):
                    return f'torch.Tensor {t.shape} of type {t.dtype} on {t.device}'
                elif isinstance(t, np.ndarray):
                    return f'numpy.array {t.shape} of type {t.dtype}'
                else:
                    return str(t)
            info = pprint.pformat({ k: _format_object(v) for k,v in it.items() })
        elif isinstance(it, (list, tuple)):
            info = f'{type(it)} of '
            info += str(list(map(_format_object, it)))
        else:
            info = it
        nl = "\n"
        return f'torchvtk.datasets.TorchDataset ({len(self)} items){nl}From {self.path}{nl}Sample:{nl}{info}'''

    def __str__(self): return repr(self)
    @staticmethod
    def CQ500(tvtk_ds_path='~/.torchvtk/', num_workers=0, **kwargs):
        ''' Get the QureAI CQ500 Dataset.
        Downloads, extracts and converts to TorchDataset if not locally available
        Find the dataset here: http://headctstudy.qure.ai/dataset
        Credits to Chilamkurthy et al. https://arxiv.org/abs/1803.05854

        Args:
            tvtk_ds_path(str, Path): Path where your torchvtk datasets shall be saved.
            num_workers (int): Number of processes used for downloading, extracting, converting
            kwargs: Keyword arguments to pass on to TorchDataset.__init__()

        Returns:
            TorchDataset: TorchDataset containing CQ500.
        '''
        path = Path(tvtk_ds_path).expanduser()
        path.mkdir(exist_ok=True)
        cq500path = path/'CQ500'
        if cq500path.exists() and len(list(filter(lambda n: n.endswith('.pt'), os.listdir(cq500path)))) > 0:
            return TorchDataset(cq500path, **kwargs)
        else:
            test_has_gdcm()
            orig_path = path/'CQ500_orig'
            print(f'Downloading CQ500 dataset to {orig_path}...')
            files = download_all(urls.cq500, orig_path, num_workers=num_workers)
            print('Extracting CQ500 dataset...')
            files = extract_all(orig_path, delete_archives=True, num_workers=num_workers)
            print(f'Converting CQ500 dataset to TorchDataset (in {cq500path})...')
            cq500_to_torch(orig_path, cq500path, num_workers=num_workers)
            print(f'Removing original CQ500 files ({orig_path})...')
            shutil.rmtree(orig_path)
            return TorchDataset(cq500path, **kwargs)

class DataLoadingWork:
    def __init__(self, device):
        self.device = device

    def __call__(self, fn):
        return _preload_dict_tensors(torch.load(str(fn)), device=self.device)

class PreloadedTorchDataset(TorchDataset):
    def __init__(self, torch_ds, device='cpu', num_workers=0, override_data=None):
        super().__init__(torch_ds.items, filter_fn=torch_ds.filter_fn, preprocess_fn=torch_ds.preprocess_fn)
        self.device = device
        if override_data is None:
            work = DataLoadingWork(device)
            self.data = pool_map(work, torch_ds.items, num_workers=num_workers)
            print(f'Preloaded TorchDataset ({self.path}) to ({device}).')
        else:
            self.data = override_data

    def __getitem__(self, i):
        if self.preprocess_fn is not None:
            return self.preprocess_fn(self.data[i])
        else:
            return self.data[i]

    def preload(self, device='cpu', num_workers=0):
        ''''This does nothing, as the dataset you're calling this on is preloaded already '''
        return self

    def tile(self, keys_to_tile, tile_sz=128, overlap=2, dim=3):
        ''' Converts the Dataset to a Tiled Dataset, drawing only parts of the data.
        Since this is a preloaded dataset, the tiling locations are fixed from the beginning and `__len__` will return the number of tiles.

        Args:
            keys_to_tile ([str]): List of strings matching the keys of the data dictionaries that need to be tiled. All must have the same shape and result in the same tiling.
            tile_sz (int/tuple of ints, optional): Size of the tiles drawn. Either int or tuple with length matching the given `dim`. Defaults to 128.
            overlap (int/tuple of its, optional): . Defaults to 2.
            dim (int, optional): Dimensionality of the data. If `tile_sz` or `overlap` is given as tuple this must match their lengths. Defaults to 3.

        Returns:
            PreloadedTiledTorchDataset: Tiling-aware `PreloadedTorchDataset`
        '''
        return PreloadedTiledTorchDataset(self, keys_to_tile,
            tile_sz=tile_sz,
            overlap=overlap,
            dim=dim)

def get_tile_locations(shape, tile_sz, overlap, dim=3):
    if isinstance(shape, torch.Size):
        max_dims = shape[-dim:]
    elif isinstance(shape, (tuple, list)) and len(shape) == dim:
        max_dims = shape
    else:
        raise Exception(f'Shape must the torch.Size or tuple/list with length {self.dim}. Got {shape} instead.')

    idxs = []
    for tile, maxd, overl in zip(tile_sz, max_dims, overlap):
        if tile is None:  # Use full available data
            idx = [0]
        else: # Tile
            end = maxd +1 - tile if maxd > tile else 0
            step = tile - overl
            idx = list(range(0, end, step)) if end > step else [0]
            if idx[-1] < end-1:
                idx = list(map(lambda x: x + (end-idx[-1])//2, idx)) # center tiles if the size is no divisible
        idxs.append(torch.LongTensor(idx))
    start = torch.unique(torch.stack(torch.meshgrid(idxs), dim=-1), dim=0)
    if None in tile_sz:
        tile_sz = list(map(lambda tm: tm[0] if tm[0] is not None else tm[1], zip(tile_sz, max_dims)))
    end   = start + torch.LongTensor(tile_sz)
    return torch.stack([start, end], dim=-2).reshape(-1, 2, 3)


class TiledTorchDataset(TorchDataset):
    def __init__(self, torch_ds, keys_to_tile, dim=3, tile_sz=128, overlap=2, return_all_tiles=False):
        super().__init__(torch_ds.items, filter_fn=torch_ds.filter_fn, preprocess_fn=torch_ds.preprocess_fn)
        self.keys_to_tile = keys_to_tile if isinstance(keys_to_tile, (tuple, list)) else (keys_to_tile,)
        self.dim = dim
        self.return_all_tiles = return_all_tiles
        self.tile_sz = tile_sz if isinstance(tile_sz, (tuple, list)) else (tile_sz,)*dim
        self.overlap = overlap if isinstance(overlap, (tuple, list)) else (overlap,)*dim
        assert len(self.tile_sz) == dim and len(self.overlap) == dim

    def __getitem__(self, i):
        item = torch.load(str(self.items[i]))
        tile = {**item}
        for k in self.keys_to_tile:
            tile_locations = self._get_tile_locations(item[k].shape)
            n_tiles = len(tile_locations)
            if 'num_tiles' in tile.keys():
                assert n_tiles == tile['num_tiles']
            tile['num_tiles'] = n_tiles
        if self.return_all_tiles: # Return all tiles from this item
            all_tiles = []
            for crop in tile_locations:
                t = clone(tile)
                t['tile_location'] = crop
                for k in self.keys_to_tile:
                    prev = [slice(None)]*(item[k].ndim - self.dim)
                    slices = prev + [slice(c[0].item(), c[1].item()) for c in crop.transpose(0, 1)]
                    t[k] = clone(item[k][tuple(slices)])
                if self.preprocess_fn is not None:
                    all_tiles.append(self.preprocess_fn(t))
                else: all_tiles.append(t)
            return all_tiles

        else: # Return only a single random tile
            tile_idx = torch.randint(0, tile['num_tiles'], (1,)).item()
            crop = tile_locations[tile_idx]
            tile['tile_location'] = crop
            for k in self.keys_to_tile:
                prev = [slice(None)]*(item[k].ndim - self.dim)
                slices = prev + [slice(c[0].item(), c[1].item()) for c in crop.transpose(0, 1)]
                tile[k] = clone(item[k][tuple(slices)])
                # if tuple(tile[k].shape[-self.dim:]) != self.tile_sz:
                #     old_dtype = tile[k].dtype
                #     tile[k] = F.interpolate(make_5d(tile[k]).float(), size=self.tile_sz).squeeze(0).to(old_dtype)
            if self.preprocess_fn is not None:
                return self.preprocess_fn(tile)
            else:
                return tile

    def _get_tile_locations(self, shape): # TODO: docstring
        return get_tile_locations(shape, self.tile_sz, self.overlap, self.dim)

    def preload(self, device='cpu', num_workers=0):
        return (TorchDataset(self.items,
                filter_fn=self.filter_fn,
                preprocess_fn=self.preprocess_fn)
                .preload(device=device, num_workers=num_workers)
                .tile(self.keys_to_tile,
                    dim=self.dim,
                    tile_sz=self.tile_sz,
                    overlap=self.overlap)
        )

    def tile(self, keys_to_tile, tile_sz=128, overlap=2, dim=3):
        ''''This does nothing, as the dataset you're calling this on is tiled already '''
        return self


class PreloadedTiledTorchDataset(PreloadedTorchDataset):
    def __init__(self, torch_ds, keys_to_tile, dim=3, tile_sz=128, overlap=2, **preload_kwargs):
        super().__init__(torch_ds, **preload_kwargs)
        self.keys_to_tile = keys_to_tile if isinstance(keys_to_tile, (tuple, list)) else (keys_to_tile,)
        self.dim = dim
        self.tile_sz = tile_sz if isinstance(tile_sz, (tuple, list)) else (tile_sz,)*dim
        self.overlap = overlap if isinstance(overlap, (tuple, list)) else (overlap,)*dim
        assert len(self.tile_sz) == dim and len(self.overlap) == dim
        self.data = torch_ds.data
        self.cum_num_tiles = [0]
        for data in self.data:
            for k in keys_to_tile:
                data['tile_locations'] = self._get_tile_locations(data[k].shape)
                n_tiles = len(data['tile_locations'])
                if 'num_tiles' in data.keys():
                    assert n_tiles == data['num_tiles']
                data['num_tiles'] = n_tiles
            self.cum_num_tiles.append(n_tiles + self.cum_num_tiles[-1])

    def __len__(self): return self.cum_num_tiles[-1]

    def __getitem__(self, i):
        item_id = bisect(self.cum_num_tiles, i) - 1
        item = self.data[item_id]
        tile_id = i - self.cum_num_tiles[item_id]
        crop = item['tile_locations'][tile_id]
        shapes = {f'{k}_shape': item[k].shape for k in self.keys_to_tile}
        tile = {
            'tile_id': tile_id,
            'item_id': item_id,
            'tile_location': crop, **shapes, **item}
        for k in self.keys_to_tile:
            prev = [slice(None)]*(item[k].ndim - self.dim)
            slices = prev + [slice(c[0].item(), c[1].item()) for c in crop.transpose(0, 1)]
            tile[k] = clone(item[k][tuple(slices)])
            # if tuple(tile[k].shape[-self.dim:]) != self.tile_sz:
            #     old_dtype = tile[k].dtype
            #     tile[k] = F.interpolate(make_5d(tile[k]).float(), size=self.tile_sz).squeeze(0).to(old_dtype)
        if self.preprocess_fn is not None:
            return self.preprocess_fn(tile)
        else:
            return tile

    def _get_tile_locations(self, shape): # TODO: docstring
        return get_tile_locations(shape, self.tile_sz, self.overlap, self.dim)


    def tile(self, keys_to_tile, tile_sz=128, overlap=2, dim=3):
        ''''This does nothing, as the dataset you're calling this on is tiled already '''
        return self

    def preload(self, device='cpu', num_workers=0):
        ''''This does nothing, as the dataset you're calling this on is preloaded already '''
        return self

# %%
