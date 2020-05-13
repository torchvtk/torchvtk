from pathlib  import Path
from argparse import ArgumentParser

import numpy as np
import torch
import dicom_numpy

from utils import *
from torch_volume_dataloaders.utils.volume_utils import normalize_hounsfield, normalize_voxel_scale, make_4d

def get_volume_gen(volume_dirs, apply_dcm_rescale=False, permute_c_contiguous=True):
    ''' Make a generator that loads volumes from a list of volume directories, `volume_dirs`.
    Returns: (volume:np.ndarray , index_to_pos_4x4:np.ndarray) '''
    rescale = None if apply_dcm_rescale else False
    def vol_gen():
        for vol_dir in volume_dirs:
            with hidden_errors():
                try:
                    vol, dicom = read_dicom_folder(vol_dir, rescale)
                    if permute_c_contiguous: # return (D, H, W) instead of (W, H, D)
                          vol = np.ascontiguousarray(vol.transpose(2,1,0))
                          vox_scl = np.array([dicom.SliceThickness, dicom.PixelSpacing[1], dicom.PixelSpacing[0]]).astype(np.float32)
                    else: vox_scl = np.array([dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]).astype(np.float32)
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

def process_volumes(vol_gen, save_path, dtype=torch.float16, normalize_voxel_scl=False, normalize_intensities=True, print_info=False):
    assert isinstance(dtype, torch.dtype)
    save_path = Path(save_path)
    if not save_path.exists(): save_path.mkdir()
    for vol, vox_scl, vol_name in vol_gen:
        vol     = make_4d(torch.FloatTensor(vol.astype(np.float32)))
        vox_scl = torch.FloatTensor(vox_scl)
        if normalize_intensities: vol = normalize_hounsfield(vol)
        if normalize_voxel_scl:
            vol = make_4d(normalize_voxel_scale(vol, vox_scl).squeeze())
            vox_scl = torch.ones(3)
        file_path = save_path/f"{vol_name.replace('-', '_')}.pt"
        torch.save({
            'vol': vol.to(dtype),
            'vox_scl': vox_scl,
            'name': vol_name,
            'dataset': 'CQ500'
        }, file_path)
        if print_info: print(f'Saved Volume {vol_name} with shape {vol.shape} ({dtype}) (Vox Scale: {vox_scl}) to {file_path}')

if __name__=='__main__':
    parser = ArgumentParser("CQ500 dataset preprocessor", description='''
        Reads the CQ500 dataset from QureAI (http://headctstudy.qure.ai/dataset)
        in the original DICOM format and converts it to serialized PyTorch tensors.
        Expected qure_path structure:
            Qure_AI_Brain_CT / subfolders for subjects / subj name / Unknown Study / folders with .dcm's
            e.g. Qure_AI_Brain_CT/CQ500-CT-43/CQ500CT43 CQ500CT43/Unknown Study/CT 2.55mm/
    ''')
    parser.add_argument('qure_path', type=str, help='Path to the Qure_AI_Brain_CT root folder.')
    parser.add_argument('save_path', type=str, help='Path to save the serialized PyTorch tensors to.')
    parser.add_argument('-min_slices', type=int, default=0,            help='Minimum number of slices to consider volume.')
    parser.add_argument('-max_slices', type=int, default=float("inf"), help='Maximum number of slices to consider volume.')
    args = parser.parse_args()

    vol_dirs = traverse_cq500_folders(args.qure_path, args.min_slices, args.max_slices)
    vol_gen = get_volume_gen(vol_dirs)
    process_volumes(vol_gen, args.save_path)
