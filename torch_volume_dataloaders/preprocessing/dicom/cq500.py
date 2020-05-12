from pathlib import Path

import numpy as np
import dicom_numpy

from .utils import *

def get_volume_gen(volume_dirs, apply_dcm_rescale=False):
    ''' Make a generator that loads volumes from a list of volume directories, `volume_dirs`.
    Returns: (volume:np.ndarray , index_to_pos_4x4:np.ndarray) '''
    rescale = None if apply_dcm_rescale else False
    def vol_gen():
        for vol_dir in volume_dirs:
            with hidden_errors():
                try:
                    vol, dcm = read_dicom_folder(vol_dir, rescale)
                    vox_scl = np.array([dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]).astype(np.float32)
                    vox_scl /= vox_scl.min()
                    vol_name = str(vol_dir.parent.parent.parent.name)
                except dicom_numpy.DicomImportException:
                    print(f'Could not load {vol_dir}')
                    continue
            yield vol, vox_scl, vol_name
    return vol_gen()

def traverse_cq500_folders(path, min_slices=1, max_slices=float("inf")):
    path = Path(path)
    flatten = lambda l: [item for sublist in l for item in sublist]
    return list(
           filter(num_slices_between(min_slices, max_slices),  # extract subdir with most files in it (highest res volume)
           flatten(
           map(   lambda p: list(p.iterdir()),                 # get list of actual volume directorie
           map(   lambda p: next(p.iterdir())/'Unknown Study', # cd into subfolders CQ500-CT-XX/Unknown Study/
           filter(lambda p: p.is_dir(),                        # Get all dirs, no files
           path.iterdir()))))))                                # Iterate over path directory
