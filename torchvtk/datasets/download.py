import requests
from pathlib import Path
from functools import partial
import torch.multiprocessing as mp
import zipfile, tarfile, shutil
from torchvtk.utils import pool_map
import tqdm

def download(url, target_folder):
    path = Path(target_folder)
    fn = url[url.rfind("/") + 1:]
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(path/fn, 'wb') as f:
            for data in r: f.write(data)

    return path/fn

def download_all(urls, target_folder, num_workers=0):
    Path(target_folder).mkdir(exist_ok=True)
    dl_fn = partial(download, target_folder=target_folder)
    if num_workers > 0:
        return pool_map(dl_fn, urls, num_workers=num_workers)

    else:
        return [download(url, target_folder) for url in tqdm.tqdm(urls)]


def untar(fn, target_dir=None, delete_archive=False):
    fn = Path(fn)
    if target_dir is None: target_dir = fn.parent
    with tarfile.open(fn, 'r:*') as f:
        f.extractall(target_dir)
    if delete_archive: fn.unlink()
    return target_dir

def unzip(fn, target_dir=None, delete_archive=False):
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
    path = Path(dir)
    if target_dir is None: target_dir = path
    target_dir.mkdir(exist_ok=True)
    zips = path.rglob('*.zip')
    zip_fn = partial(unzip, target_dir=target_dir, delete_archive=delete_archives)
    tars = path.rglob('*.tar.gz')
    tar_fn = partial(untar, target_dir=target_dir, delete_archive=delete_archives)

    unzipped = pool_map(zip_fn, zips, num_workers=num_workers)
    untarred = pool_map(tar_fn, tars, num_workers=num_workers)

    if delete_archives: shutil.rmtree(path)
