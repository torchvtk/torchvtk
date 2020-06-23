from time import time

import torch
from torch.utils.data import Dataset
import numpy as np

from tqdm import trange

import matplotlib.pyplot as plt

__all__ = ['print_results', 'run_benchmark']

def noop(a): return a

def print_results(result_dict):
    ''' Prints dictionary in a readable way (trimming long lists, reducing tensors to shape and dtype). '''
    for k, v in result_dict.items():
        if   isinstance(v, (list, tuple)):
            print(f'{k}: List ({len(v)}): {v[:10]}....')
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(f'{k}: Tensor ({v.shape}, dtype={v.dtype}')
        else:
            print(f'{k}: {v}')

def run_benchmark(dataset, pct=None, preprocess_fn=noop, print_plot=True, save_plot=None):
    ''' Benchmarks a Dataset by loading `pct` percent of the dataset, measuring performance and plotting results
    Args:
        dataset (torch.utils.data.Dataset): A PyTorch dataset
        pct (float in [0,1]): Percentage of the dataset to benchmark
        preprocess_fn (function): A function that is applied to the object returned by the dataset (included in time measurements)
        print_plot (bool): Whether to print a plot of the results
        save_plot (string, Path): Path where the result plot is saved to. Include a file extension!
    Returns:
        A dictionary containing the `times`, tensor `sizes` and metrics (`average_time`, `max_time`, `total_time`).
    '''
    num_items = len(dataset)
    if pct is not None and pct > 0.0 and pct < 1.0:
        num_items = int(pct * len(dataset))
    times = []
    sizes = []
    for i in trange(num_items):
        before = time()
        it = preprocess_fn(dataset[i])
        after = time()
        if torch.is_tensor(it): sizes.append(it.element_size() * it.nelement() / 1e6)
        else:                   sizes.append(1.0)
        times.append(after - before)
    # Metrics
    sizes, times = np.array(sizes), np.array(times)
    if (sizes == 1.0).all(): print('Warning: Preprocess function idd not return a tensor, all sizes set to 1MB.')
    avg_time = np.mean(times)
    max_time = np.max(times)
    total_time = np.sum(times)
    time_per_mb = times / sizes
    # Plots
    if print_plot or save_plot is not None:
        idx = np.argsort(sizes)
        size_sorted = sizes[idx]
        time_sorted = times[idx]
        fig, axs = plt.subplots(2,1)

        axs[0].hist(times, color='b')
        axs[0].set_xlabel("Time in Seconds")
        axs[0].set_ylabel("Frequency")
        axs[1].plot(size_sorted, time_sorted)
        axs[1].set_xlabel("Size in MB")
        axs[1].set_ylabel("Time to load")

        if save_plot is not None: fig.savefig(save_plot)
        if print_plot:            fig.show()

    return {'average_time': avg_time,
            'max_time':     max_time,
            'total_time':   total_time,
            'time_per_mb':  time_per_mb,
            'times':        times,
            'sizes':        sizes}
