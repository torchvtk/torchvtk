import numpy as np
import dicom_numpy

from .utils import *

def get_volume_gen(volume_dirs, rescale=None, tf_pts=None):
    ''' Make a generator that loads volumes from a list of volume directories, `volume_dirs`.
    Returns: (volume:np.ndarray , index_to_pos_4x4:np.ndarray) '''
    def vol_gen():
        for vol_dir in volume_dirs:
            with hidden_errors():
                try:
                    vol, dcm = read_dicom_folder(vol_dir, rescale)
                    vox_scl = np.array([dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]).astype(np.float32)
                    vox_scl /= vox_scl.min()
                    vol_name = str(vol_dir.parent.parent.parent.name)
                    if tf_pts is None:
                        peaks  = get_histogram_peaks(normalized_vol)
                        tf_pts = get_trapezoid_tf_points_from_peaks(peaks)
                except dicom_numpy.DicomImportException:
                    print(f'Could not load {vol_dir}')
                    continue
            yield vol, tf_pts, vox_scl, vol_name
    return vol_gen()
