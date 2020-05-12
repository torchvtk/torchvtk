import os
import sys
from contextlib import contextmanager
from pathlib    import Path

import numpy as np

import pydicom
import dicom_numpy

__all__ = ['hidden_prints', 'hidden_errors', 'read_dicom_folder', 'get_largest_dir',
    'num_slices_between', 'get_volume_dirs']

@contextmanager
def hidden_prints():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def hidden_errors():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def read_dicom_folder(dicom_folder, rescale=None):
    ''' Reads all .dcm files in `dicom_folder` and merges them to one volume

    Returns:
        The volume and the affine transformation from pixel indices to xyz coordinates
    '''
    dss = [pydicom.dcmread(str(dicom_folder/dcm)) for dcm in os.listdir(dicom_folder) if dcm.endswith('.dcm')]
    vol, mat = dicom_numpy.combine_slices(dss, rescale)
    return vol, dss[0]

def get_largest_dir(dirs, minsize=100):
    ''' Returns the dir with the most files from `dirs`'''
    m = max(dirs, key=lambda d: len(os.listdir(d)) if os.path.isdir(d) else 0)
    if len(os.listdir(m)) >= minsize: return m
    else: return None

def num_slices_between(minz, maxz):
    def _comp(path):
        num_slices = len(os.listdir(path))
        return minz <= num_slices and num_slices <= maxz
    return _comp

def get_volume_dirs(path):
    path = Path(path)
    return list(
        filter(lambda p: p is not None,
        map(   get_largest_dir,                             # extract subdir with most files in it (highest res volume)
        map(   lambda p: list(p.iterdir()),                 # get list of actual volume directorie
        map(   lambda p: next(p.iterdir())/'Unknown Study', # cd into subfolders CQ500-CT-XX/Unknown Study/
        filter(lambda p: p.is_dir(),                        # Get all dirs, no files
        path.iterdir())))))                                 # Iterate over path directory
)
