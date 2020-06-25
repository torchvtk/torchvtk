from argparse import ArgumentParser
from pathlib  import Path

import torch

from torchvtk.datasets.hdf5_dataset import H5DatasetReopen, H5DatasetOpenOnce
from torchvtk.benchmarks.benchmark_dataset import run_benchmark, print_results

if __name__ == '__main__':
    parser = ArgumentParser('Benchmark CQ500 loading', description='''
        Benchmarks loading the CQ500 dataset as a HDF5 Dataset
    ''')
    parser.add_argument('ds_path', type=str, help='Path to the H5Dataset')
    parser.add_argument('-plot',   type=str, default=None, help='Path to save result plot (Will save Reopen and OpenOnce separate automatically)')
    parser.add_argument('-pct',    type=float, default=None, help='Percentage of the dataset to benchmark. Must be in [0,1]. Defaults to full dataset')
    parser.add_argument('-open',   type=str, default='once', help='Whether to open the HDF5 file on __getitem__ ("reopen") or only once ("once"). Defaults to once')
    parser.add_argument('--noplot', action='store_false', help="Don't print plot of results")
    args = parser.parse_args()

    if args.open.lower() == 'reopen':
          ds_cls = H5DatasetReopen
    else: ds_cls = H5DatasetOpenOnce

    print(f'======== Results for H5Dataset ({ds_cls}) {ds_path} ============')
    ds = ds_cls(args.ds_path, preprocess_fn=lambda d: d['vol'])
    results = run_benchmark(ds, save_plot=args.plot, pct=args.pct, print_plot=args.noplot)
    print_results(results)
