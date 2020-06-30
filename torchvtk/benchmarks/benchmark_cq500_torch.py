#%%
from argparse import ArgumentParser
from pathlib  import Path

import torch

from torchvtk.datasets.torch_dataset import TorchDataset
from torchvtk.benchmarks.benchmark_dataset import run_benchmark

if __name__ == '__main__':
    parser = ArgumentParser('Benchmark CQ500 loading', description='''
        Benchmarks loading the CQ500 dataset as a TorchDataset
    ''')
    parser.add_argument('ds_path', type=str, help='Path to the TorchDataset')
    parser.add_argument('-plot',   type=str, default=None, help='Path to save result plot')
    parser.add_argument('-pct',    type=float, default=None, help='Percentage of the dataset to benchmark. Must be in [0,1]')
    parser.add_argument('--noplot', action='store_false', help="Don't print plot of results")
    args = parser.parse_args()

    ds = TorchDataset(args.ds_path, preprocess_fn=lambda d: d['vol'])

    results = run_benchmark(ds, save_plot=args.plot, pct=args.pct, print_plot=args.noplot)
    print(results)
