#%%
import torch
from torch.utils.data import IterableDataset
import torch.multiprocessing as mp
import numpy as np

from itertools import cycle
from functools import partial
import time

from torch_volume_dataloaders.datasets.torch_dataset import TorchDataset

#%%
def noop(a, *args, **kwargs): return a

def dict_collate_fn(items, key_filter_fn=None, stack_tensors=True):
    keys = items[0].keys()
    if key_filter_fn is not None:
        keys = filter(key_filter_fn, keys)
    batch = {}
    for k in keys:
        vals = [it[k] for it in items]
        if stack_tensors and torch.is_tensor(vals[0]):
              batch[k] = torch.stack(vals)
        else: batch[k] = vals
    return batch

def _share_mem(d):
    for t in d.values():
        if torch.is_tensor(t): t.share_memory_()

def load_always(ds, queue, q_maxlen, tfm=noop):
    while True:
        idxs = torch.randperm(len(ds))
        for i in idxs:
            d = tfm(ds[i])
            _share_mem(d)
            if len(queue) >= q_maxlen: queue.pop()
            queue.insert(0, d)

def load_onsample(ds, queue, q_maxlen, sample_event, tfm=noop):
    while True:
        idxs = torch.randperm(len(ds))
        for i in idxs:
            d = tfm(ds[i])
            _share_mem(d)
            sample_event.wait()
            if len(queue) >= q_maxlen-1: sample_event.clear()
            queue.insert(0, d)

class TorchQueueDataset(IterableDataset):
    def __init__(self, torch_ds, mode='onsample', fill_interval=None, num_workers=1, q_maxlen=None, ram_use=0.75,
        wait_fill=True, sample_tfm=noop, bs=1, collate_fn=dict_collate_fn):
        '''
        Args:
            torch_ds (TorchDataset): A TorchDataset to be used for queueing
            mode (string): Queue filling mode.
                - 'onsample' refills the queue after it got sampled
                - 'always' keeps refilling the queue as fast as possible
                - 'interval' refills the queue regularly. Set fill_interval
            fill_interval (float): Time intervals in seconds between queue fillings
            num_workers (int): Number of threads loading in data
            q_maxlen (int): Set queue size. Overrides `ram_use`
            ram_use (float): Fraction of available system memory to use for queue. Default is 75%
            wait_fill (int, bool): Boolean whether queue should be filled on init or Int to fill the queue at least with a certain amount of items
            sample_tfm (Transform, function): Applicable transform (receiving and producing a dict) that is applied upon sampling from the queue
            bs (int): Batch Size
            collate_fn (function): Collate Function to merge items to batches. Default assumes dictionaries (like from TorchDataset) and stacks all tensors, while collecting non-tensors in a list
        '''
        self.bs = bs
        self.collate_fn = collate_fn
        # Split dataset into num_workers sub-datasets
        self.items = list(map(list, np.array_split(torch_ds.items, num_workers)))
        self.datasets = [TorchDataset(items, preprocess_fn=torch_ds.preprocess_fn) for items in self.items]
        self.sample_tfm = sample_tfm
        self.q_maxlen = q_maxlen if q_maxlen is not None else self._get_queue_sz(ram_use)
        self.manager = mp.Manager() # Manager for shared resources
        self.mode = mode # Set worker functions for dataloading mode
        if   mode == 'onsample': # Pops and adds a new item when sampled
            self.sample_event = mp.Event()
            self.sample_event.set()
            worker_fn = partial(load_onsample, sample_event=self.sample_event)
        elif mode == 'always': # Pops and adds new items as fast das the dataloaders can
            worker_fn = partial(load_always)
        else: raise Exception(f'Invalid queue filling mode: {mode}')
        self.queue   = self.manager.list() # Shared list of tensors
        self.workers = [ mp.Process( # Start worker processes
            target = worker_fn,
            args   = (ds, self.queue, self.q_maxlen), # Dataset, Queue, Queue Size
            daemon = True
        ) for ds in self.datasets]
        # Start Jobs & possibly Wait
        for w in self.workers: w.start()
        if int(wait_fill) > 0:
            wait_for = min(self.q_maxlen, wait_fill) if isinstance(wait_fill, int) else self.q_maxlen
            self.wait_fill_queue(fill_atleast=wait_for)

    @property
    def qsize(self): return len(self.queue)

    def batch_generator(self):
        ''' Generator for sampling the queue. '''
        while True:
            idxs = torch.randperm(len(self.queue))[:self.bs]
            samples = [self.sample_tfm(self.queue[i]) for i in idxs]
            if self.mode == 'onsample':
                self.queue.pop()
                self.sample_event.set()
            if self.bs == 1: yield samples[0]
            else:            yield self.collate_fn(samples)

    def __iter__(self): return iter(self.batch_generator())

    def wait_fill_queue(self, fill_atleast=None, timeout=60, polling_interval=0.25):
        timed_out = time.time() + timeout
        while time.time() < timed_out:
            if (self.qsize >= self.q_maxlen or
               (fill_atleast is not None and self.qsize >= fill_atleast)):
                return
            else: time.sleep(polling_interval)
        print(f'Warning: Queue is not filled, but timeout of {timeout}s was reached!')

    def _get_queue_sz(self, ram_use):
        ''' Determines a queue size from available system memory '''
        return 32 # TODO: replace this dummy, implement actual logic
