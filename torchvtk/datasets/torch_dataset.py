#%%
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import numpy as np

from pathlib import Path
from functools import partial
import shutil, os, pprint
import torchvtk.datasets.urls as urls
from torchvtk.datasets.download import download_all, extract_all
from torchvtk.converters.dicom import cq500_to_torch, test_has_gdcm
from torchvtk.utils import pool_map

def _preload_dict_tensors(it, device='cpu'):
    for k, v in it.items():
        if torch.is_tensor(v): it[k] = v.to(device)

class DatasetWork:
    def __init__(self, items, target_path, process_fn):
        self.items = items
        self.target_path = target_path
        self.process_fn = process_fn
    def work_fn(self, i):
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
        if  isinstance(ds_files, (str, Path)):
            self.path = Path(ds_files)
            assert self.path.is_dir()
            items = self.path.rglob('*.pt')
            if filter_fn is not None:
                items = filter(filter_fn, items)
            self.items = list(items)
        elif isinstance(ds_files, (list, tuple)):
            for f in ds_files:
                assert Path(f).is_file() and Path(f).suffix == '.pt'
            self.path = ds_files[0].parent
            if filter_fn is not None:
                  self.items = list(filter(filter_fn, ds_files))
            else: self.items = list(ds_files)

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
        pool_map(work.work_fn, [i for i in range(len(self))], num_workers=num_workers)

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
            TorchDataset: New TorchDataset using the preloaded data.
        '''
        self.data = [self[i] for i in range(len(self))]
        pool_map(partial(_preload_dict_tensors, device=device), self.data, num_workers=num_workers)
        def new_get(i):
            print('Retrieving from cache: ', i)
            return self.data[i]
        self.__getitem__ = new_get
        return self

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
