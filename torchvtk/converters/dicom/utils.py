import os
import sys
from contextlib import contextmanager
from pathlib    import Path

import numpy as np

try:
    _has_gdcm = False
    import pydicom
    _has_gdcm = pydicom.pixel_data_handlers.gdcm_handler.is_available()
    import dicom_numpy
except ImportError:
    _has_dicom = False
else:
    _has_dicom = True

__all__ = ['requires_gdcm', 'requires_dicom', 'test_has_dicom', 'test_has_gdcm',
    'hidden_prints', 'hidden_errors', 'read_dicom_folder', 'get_largest_dir', 'num_slices_between']

def requires_gdcm(func):
    def _dummy(*args, **kwargs):
        raise ImportError('You need to install gdcm with pydicom: https://pydicom.github.io/pydicom/stable/tutorials/installation.html#installing-gdcm')
    if _has_gdcm: return func
    else:         return _dummy

def requires_dicom(func):
    def _dummy(*args, **kwargs):
        raise ImportError('pydicom and dicom_numpy are required.')
    if _has_dicom: return func
    else:          return _dummy

@requires_dicom
def test_has_dicom():
    'Tests if pydicom and dicom_numpy are installed.'
    pass

@requires_gdcm
def test_has_gdcm():
    'Tests if gdcm (and thus pydicom) is installed.'
    pass

@contextmanager
def hidden_prints():
    ''' Context manager that blocks all prints within. '''
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def hidden_errors():
    ''' Context manager that blocks all error prints within. '''
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@requires_dicom
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
    ''' Returns a function that checks the number of slices of a DICOM dir by counting its .dcm files. To be used in filter '''
    def _comp(path):
        num_slices = len(os.listdir(path))
        return minz <= num_slices and num_slices <= maxz
    return _comp


# IF MODIFIED ADAPT __all__ variable on top!
