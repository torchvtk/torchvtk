import torch.multiprocessing as mp
import tqdm

def pool_map(fn, data, num_workers=0, dlen=None, title=None):
    ''' Multithreaded map function that displays a progress bar

    Args:
        fn (function): Function to be applied to the elements in `data`
        data (iterable): Iterable on which the function `fn` is applied.
        num_workers (int): Number of threads to do the computation
        dlen (int): A way to supply the length of `data` separately (to display in progress bar)
        title (str): Title to be displayed next to the progress bar

    Returns:
        A list of results [fn(data[0]), .... fn(data[-1])]
    '''
    result = []
    if num_workers > 0:
        n = len(data) if hasattr(data, '__len__') else dlen
        desc = title if title is not None else fn.__name__ if hasattr(fn, '__name__') else None
        with mp.Pool(num_workers) as p:
            with tqdm.tqdm(total=n, desc=desc) as bar:
                for r in p.imap(fn, data):
                    result.append(r)
                    bar.update()
    else: result = [fn(d) for d in tqdm.tqdm(data)]
    return result

def pool_map_uo(fn, data, num_workers=0, dlen=None):
    ''' Multithreaded unordered map function that displays a progress bar

    Args:
        fn (function): Function to be applied to the elements in `data`
        data (iterable): Iterable on which the function `fn` is applied.
        num_workers (int): Number of threads to do the computation
        dlen (int): A way to supply the length of `data` separately (to display in progress bar)

    Returns:
        A list of results [fn(data[0]), .... fn(data[-1])]
    '''
    result = []
    if num_workers > 0:
        n = len(data) if hasattr(data, '__len__') else dlen
        desc = fn.__name__ if hasattr(fn, '__name__') else None
        with mp.Pool(num_workers) as p:
            with tqdm.tqdm(total=n, desc=desc) as bar:
                for r in p.imap_unordered(fn, data):
                    result.append(r)
                    bar.update()
    else: result = [fn(d) for d in tqdm.tqdm(data)]
    return result
