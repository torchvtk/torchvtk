#%%
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.multiprocessing as mp
import numpy as np

from itertools import cycle
from functools import partial
from pathlib   import Path
from numbers   import Number
import time, math, psutil, os, numbers
from collections import defaultdict

from torchvtk.datasets import TorchDataset

#%%
def noop(a, *args, **kwargs): return a

def dict_collate_fn(items, key_filter=None, stack_tensors=True, convert_np=True, convert_numbers=True, warn_when_unstackable=True):
    ''' Collate function for dictionary data

    This stacks tensors only if they are stackable, meaning they are of the same shape.

    Args:
        items (list): List of individual items for a Dataset
        key_filter (list of str or callable, optional): A list of keys to filter the dict data. Defaults to None.
        stack_tensors (bool, optional): Wether to stack dict entries of type torch.Tensors. Disable if you have unstackable tensors. They will be stacked as a list. Defaults to True.
        convert_np (bool, optional): Convert NumPy arrays to torch.Tensors and stack them. Defaults to True.
        convert_numbers (bool, optional): Converts standard Python numbers to torch.Tensors and stacks them. Defaults to True.
        warn_when_unstackable (bool, list of str, optional): If True, prints a warning when a set of Tensors is unstackable. You can also specify a list of keys for which to print the warning. Defaults to True.

    Returns:
        dict: One Dictionary with tensors stacked
    '''
    keys = items[0].keys()
    if isinstance(key_filter, (list,tuple)):
        keys = key_filter
    elif hasattr(key_filter, "__call__"):
        keys = filter(key_filter, keys)

    batch = {}
    for k in keys:
        vals = [it[k] for it in items]
        if convert_np and isinstance(vals[0], np.ndarray):
            vals = list(map(torch.from_numpy, vals))
        if convert_numbers and isinstance(vals[0], Number):
            batch[k] = torch.tensor(vals)
            continue
        if stack_tensors and torch.is_tensor(vals[0]):
            shapes = list(map(lambda a: a.shape, vals))
            stackable = shapes.count(shapes[0]) == len(shapes)
            if stackable:
                batch[k] = torch.stack(vals)
            else:
                if (isinstance(warn_when_unstackable, bool)          and      warn_when_unstackable) or \
                   (isinstance(warn_when_unstackable, (tuple, list)) and k in warn_when_unstackable):
                    print(f'Warning: dict_collate_fn() could not stack tensors! Shapes: {shapes}')
                batch[k] = vals
        else: batch[k] = vals
    return batch

def _share_mem(d):
    for t in d.values():
        if torch.is_tensor(t): t.share_memory_()

def load_always(ds, queue, q_maxlen, lock, tfm=noop):
    ''' Worker job used to fill queue async as fast as possible. '''
    while True:
        idxs = torch.randperm(len(ds))
        for i in idxs:
            d = tfm(ds[i])
            _share_mem(d)
            lock.acquire()
            queue.insert(0, d)
            if len(queue) > q_maxlen: queue.pop()
            lock.release()

def load_onsample(ds, queue, q_maxlen, lock, sample_event, tfm=noop):
    ''' Worker job used to fill queue on sampling. '''
    while True:
        idxs = torch.randperm(len(ds))
        for i in idxs:
            d = tfm(ds[i])
            _share_mem(d)
            lock.acquire()
            if sample_event.wait():
                if len(queue) >= q_maxlen-1: sample_event.clear()
                queue.insert(0, d)
            lock.release()

class TorchQueueDataset(IterableDataset):
    def __init__(self, torch_ds, epoch_len=1000, mode='onsample', num_workers=1, q_maxlen=None, ram_use=0.5,
        wait_fill=True, wait_fill_timeout=60, sample_tfm=noop, batch_tfm=noop, bs=1, collate_fn=dict_collate_fn, log_sampling=False,
        avg_item_size=None, preprocess_fn=None, filter_fn=None):
        ''' An iterable-style dataset that caches items in a queue in memory.

        Args:
            torch_ds (TorchDataset, str,Path): A TorchDataset to be used for queueing or path to the dataset on disk
            mode (string): Queue filling mode.
                - 'onsample' refills the queue after it got sampled
                - 'always' keeps refilling the queue as fast as possible
            num_workers (int): Number of threads loading in data
            q_maxlen (int): Set queue size. Overrides `ram_use`
            ram_use (float): Fraction of available system memory to use for queue or memory budget in MB (>1.0). Default is 75%
            avg_item_size (float, torch.Tensor): Example tensor or size in MB
            wait_fill (int, bool): Boolean whether queue should be filled on init or Int to fill the queue at least with a certain amount of items
            wait_fill_timeout (int,float): Time in seconds until wait_fill timeouts. Default is 60s
            sample_tfm (Transform, function): Applicable transform (receiving and producing a dict) that is applied upon sampling from the queue
            batch_tfm (Transform, function):  Transforms to be applied on batches of items
            preprocess_fn (function): Override preprocess_fn from given torch_ds
            filter_fn (function): Filters filenames to load, like TorchDataset. Only used if `torch_ds` is a path to a dataset.
            bs (int): Batch Size
            collate_fn (function): Collate Function to merge items to batches. Default assumes dictionaries (like from TorchDataset) and stacks all tensors, while collecting non-tensors in a list
        '''
        self.bs = bs
        self.avg_item_size = avg_item_size
        self.collate_fn = collate_fn
        self.epoch_len = epoch_len
        self.log_sampling = log_sampling
        self.sample_dict = defaultdict(int)
        # Split dataset into num_workers sub-datasets
        if isinstance(torch_ds, TorchDataset):
            self.items = list(map(list, np.array_split(torch_ds.items, num_workers)))
            if preprocess_fn is None: preprocess_fn = torch_ds.preprocess_fn
        else:
            fns = Path(torch_ds).rglob('*.pt')
            if filter_fn is not None: fns = filter(filter_fn, fns)
            self.items = list(map(list, np.array_split(list(fns), num_workers)))
        self.datasets = [TorchDataset(items, preprocess_fn=preprocess_fn) for items in self.items]
        self.sample_tfm = sample_tfm
        self.batch_tfm  = batch_tfm
        self.q_maxlen = q_maxlen if q_maxlen is not None else self._get_queue_sz(ram_use, file_list=self.items[0])
        self.manager = mp.Manager() # Manager for shared resources
        self.queue   = self.manager.list() # Shared list of tensors
        self.lock    = self.manager.Lock()
        self.mode = mode # Set worker functions for dataloading mode
        if   mode == 'onsample': # Pops and adds a new item when sampled
            worker_fn = load_onsample
            self.sample_event = self.manager.Event()
            args = (self.queue, self.q_maxlen, self.lock, self.sample_event)
        elif mode == 'always':   # Pops and adds new items as fast das the dataloaders can
            worker_fn = load_always
            args = (self.queue, self.q_maxlen, self.lock)
        else: raise Exception(f'Invalid queue filling mode: {mode}')
        self.workers = [mp.Process(target=worker_fn, args=(ds,)+args, daemon=True) for ds in self.datasets]

        # Start Jobs & possibly Wait
        if self.mode == 'onsample': self.sample_event.set()
        for w in self.workers: w.start()
        if int(wait_fill) > 0:
            wait_for = min(self.q_maxlen, wait_fill) if isinstance(wait_fill, int) else self.q_maxlen
            self.wait_fill_queue(fill_atleast=wait_for, timeout=wait_fill_timeout)

    @property
    def qsize(self):
        ''' Current Queue length '''
        return len(self.queue)

    def get_dataloader(self, **kwargs):
        '''
        Returns:
            torch.utils.data.DataLoader: A dataloader that uses the batched sampling of the queue with appropriate collate_fn and batch_size. '''
        return DataLoader(self, batch_size=1, collate_fn=lambda it: it[0], **kwargs)

    def batch_generator(self):
        ''' Generator for sampling the queue. This makes use of the object attributes bs (batch size) and the collate function

        Returns:
            Generator that samples randomly samples batches from the queue.
        '''
        assert self.qsize >= self.bs, "Queue is not full enough to sample a single batch!"
        while True:
            idxs = torch.randperm(min(len(self.queue), self.q_maxlen))[:self.bs]
            samples = [self.sample_tfm(self.queue[i]) for i in idxs]
            if self.log_sampling:
                for s in samples: self.sample_dict[s['name']] += 1
            if self.mode == 'onsample' and self.qsize >= self.q_maxlen:
                self.queue.pop()
                self.sample_event.set()
            yield self.batch_tfm(self.collate_fn(samples))

    def __iter__(self): return iter(self.batch_generator())

    def __repr__(self):
        nl = "\n"
        nw = len(self.workers)
        return f'torchvtk.datasets.TorchQueueDataset (Queue Length {self.qsize}/{self.q_maxlen}){nl}{nw} Workers fetching {self.mode} from {nw} Datasets like:{nl}{str(self.datasets[0])}'

    def __str__(self): return repr(self)
    def wait_fill_queue(self, fill_atleast=None, timeout=60, polling_interval=0.25):
        ''' Waits untill the queue is filled (`fill_atleast`=None) or until filled with at least `fill_atleast`. Timeouts.

        Args:
            fill_atleast (int): Waits until queue is at least filled with so many items.
            timeout (Number): Time in seconds before this method terminates regardless of the queue size
            polling_interval (Number): Time in seconds how fast the queue size is polled while waiting.
        '''
        timed_out = time.time() + timeout
        while time.time() < timed_out:
            if (self.qsize >= self.q_maxlen or
               (fill_atleast is not None and self.qsize >= fill_atleast)):
                return
            else: time.sleep(polling_interval)
        print(f'Warning: Queue is not filled ({self.qsize} / {self.q_maxlen}), but timeout of {timeout}s was reached!')

    def _get_queue_sz(self, ram_use, file_list):
        ''' Determines a queue size from available system memory.

        Args:
            ram_use (float): Percentage of available system memory to use or memory budget in MB
            file_list ([Path]): List of paths. Is used to determine average item size.

        Returns:
            int: Suggested queue length
        '''
        if ram_use <= 1.0:
              mem_budget = psutil.virtual_memory().available * ram_use / 1e6
        else: mem_budget = ram_use
        if self.avg_item_size is not None:
            if torch.is_tensor(self.avg_item_size):
                avg_sz = self.avg_item_size.element_size() * self.avg_item_size.nelement() / 1e6
            elif isinstance(self.avg_item_size, numbers.Number):
                avg_sz = self.avg_item_size
            else: raise Exception(f'Invalid average item size given: {self.avg_item_size}')
        else:
            file_szs = torch.tensor(list(map(lambda p: p.stat().st_size, file_list))) / 1e6
            avg_sz = file_szs.mean().item()
        qlen = math.floor(mem_budget / avg_sz)
        print(f'''Automatic Queue Length for average Item size {avg_sz:.1f}MB:
Suggested Queue Length: {qlen}, which would take on average {qlen * avg_sz:.1f}MB memory (Budget: {mem_budget:.1f}MB).''')
        return qlen
