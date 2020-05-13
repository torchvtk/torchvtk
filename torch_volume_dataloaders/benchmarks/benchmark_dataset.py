from time import time

import torch
from torch.utils.data import Dataset
import numpy as np

import matplotlib.pyplot as plt

__all__ = ['run_benchmark']

def noop(*args, **kwargs): pass

def run_benchmark(dataset, pct=None, preprocess_fn=noop, print_plot=True, save_plot=None):
    num_items = len(dataset)
    if pct is not None and pct > 0.0 and pct < 1.0:
        num_items = int(pct * len(dataset))
    times = []
    sizes = []
    for i in range(num_items):
        before = time()
        it = dataset[i]
        preprocess_fn(it)
        after = time()
        sizes.append(it.element_size() * it.nelement() / 1e6)
        times.append(after - before)
    # Metrics
    sizes, times = np.array(sizes), np.array(times)
    avg_time = np.mean(times)
    max_time = np.max(times)
    total_time = np.sum(times)
    # Plots
    if print_plot:
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
        if save_plot is not None:
            fig.savefig(save_plot)

    return {'average_time': avg_time,
            'max_time':     max_time,
            'total_time':   total_time,
            'times':        times,
            'sizes':        sizes}
