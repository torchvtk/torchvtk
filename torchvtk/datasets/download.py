import requests
from pathlib import Path
from functools import partial
import torch.multiprocessing as mp
import zipfile, tarfile, shutil
from torchvtk.utils import pool_map
import tqdm

def download(url, target_folder):
    " Downloads a file from `url` into the `target_folder`"
    path = Path(target_folder)
    fn = url[url.rfind("/") + 1:]
    if (path/fn).exists(): return path/fn
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(path/fn, 'wb') as f:
            for data in r: f.write(data)

    return path/fn

def download_all(urls, target_folder, num_workers=0):
    ''' Downloads all files to a `target_folder` using `num_workers` threads.
    Args:
        urls ([str]): List of strings containing URLs to files
        target_folder (str, Path): Directory where the downloaded files are saved
    Returns: List of paths to the downloaded items
    '''
    Path(target_folder).mkdir(exist_ok=True)
    dl_fn = partial(download, target_folder=target_folder)
    if num_workers > 0:
        return pool_map(dl_fn, urls, num_workers=num_workers)

    else:
        return [download(url, target_folder) for url in tqdm.tqdm(urls)]


def untar(fn, target_dir=None, delete_archive=False):
    " Extracts a .tar.gz archive to `target_dir`"
    fn = Path(fn)
    if target_dir is None: target_dir = fn.parent
    with tarfile.open(fn, 'r:*') as f:
        f.extractall(target_dir)
    if delete_archive: fn.unlink()
    return target_dir

def unzip(fn, target_dir=None, delete_archive=False):
    " Extracts a .zip archive to `target_dir`"
    fn = Path(fn)
    if target_dir is None: target_dir = fn.parent
    try:
        with zipfile.ZipFile(fn, 'r') as f:
            f.extractall(target_dir)
        if delete_archive: fn.unlink()
    except zipfile.BadZipFile:
        print(f'{fn} is not a .zip')
    return target_dir

def extract_all(dir, target_dir=None, num_workers=0, delete_archives=False):
    ''' Extracts all tarball and zip archives in a given `dir`

    Args:
        dir (str, Path): Directory with .zip/.gz files to extract
        target_dir (str, Path, optional): Directory to extract the files to. Defaults to parent directory.
        num_workers (int, optional): Number of threads to use for extracting. Defaults to 0.
        delete_archives (bool, optional): Whether to delete ther archives after unpacking. Defaults to False.
    '''
    path = Path(dir)
    if target_dir is None: target_dir = path
    target_dir.mkdir(exist_ok=True)
    zips = path.rglob('*.zip')
    zip_fn = partial(unzip, target_dir=target_dir, delete_archive=delete_archives)
    tars = path.rglob('*.tar.gz')
    tar_fn = partial(untar, target_dir=target_dir, delete_archive=delete_archives)

    unzipped = pool_map(zip_fn, zips, num_workers=num_workers)
    untarred = pool_map(tar_fn, tars, num_workers=num_workers)
