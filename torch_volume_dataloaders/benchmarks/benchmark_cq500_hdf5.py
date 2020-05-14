from argparse import ArgumentParser
from pathlib  import Path

import torch

from torch_volume_dataloaders.datasets.hdf5_dataset import H5DatasetReopen, H5DatasetOpenOnce
from torch_volume_dataloaders.benchmarks.benchmark_dataset import run_benchmark, print_results

if __name__ == '__main__':
    parser = ArgumentParser('Benchmark CQ500 loading', description='''
        Benchmarks loading the CQ500 dataset as a HDF5 Dataset
    ''')
    parser.add_argument('ds_path', type=str, help='Path to the H5Dataset')
    parser.add_argument('-plot',   type=str, default=None, help='Path to save result plot')
    parser.add_argument('-pct',    type=float, default=None, help='Percentage of the dataset to benchmark. Must be in [0,1]')
    parser.add_argument('--noplot', action='store_false', help="Don't print plot of results")
    args = parser.parse_args()

    pp = Path(args.plot)
    print('======== Results for Dataset that reopens file on __get__ ============')
    reopen_pp = pp.parent / (pp.stem + '_reopen' + pp.suffix)
    ds_reopen = H5DatasetReopen(args.ds_path, preprocess_fn=lambda d: d['vol'])
    reopen_results = run_benchmark(ds_reopen, save_plot=reopen_pp, pct=args.pct, print_plot=args.noplot)
    print_results(reopen_results)

    print('======== Results for Dataset that opens file only once ============')
    openonce_pp = pp.parent / (pp.stem + '_openonce' + pp.suffix)
    ds_openonce = H5DatasetOpenOnce(args.ds_path, preprocess_fn=lambda d: d['vol'])
    openonce_results = run_benchmark(ds_openonce, save_plot=openonce_pp, pct=args.pct, print_plot=args.noplot)
    print_results(openonce_results)
