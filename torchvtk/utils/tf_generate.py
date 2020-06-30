import numpy as np
import torch

# Persistent Homology peak extraction

class Peak:
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return seq[self.born] if self.died is None else seq[self.born] - seq[self.died]

def get_persistent_homology(seq):
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)


def color_generator():
    ''' Generates distinguishable colors, compare
    http://alumni.media.mit.edu/~wad/color/numbers.html
    '''
    colors = np.array([
        [173, 35, 35],
        [42, 75, 215],
        [29, 105, 20],
        [129, 74, 25],
        [129, 38, 192],
        [160, 160, 160],
        [129, 197, 122],
        [157, 175, 255],
        [41, 208, 208],
        [255, 146, 51],
        [255, 238, 51],
        [255, 205, 243],
        [255, 255, 255]
    ], dtype=np.float32) / 255.0
    for color in colors:
        yield color

def get_histogram_peaks(data, bins=1000, skip_outlier=True):
    vals, ranges = np.histogram(data, bins)
    peaks = get_persistent_homology(vals)
    ret = np.array(list(map(lambda p: (
        (ranges[p.born] + ranges[p.born+1])/2.0,    # intensity value
        p.get_persistence(vals)), peaks # persistence for peak importance
        )))
    return np.stack([ret[:, 0], ret[:, 1] / peaks[0].get_persistence(vals)], axis=1)

def overlaps_trapeze(trap, ts):
    for t in ts:
        if trap[0,0] < t[5,0] and trap[5,0] > t[0,0]: return True
    return False

def includes_maxvalue(trap, vol=None):
    return trap[5, 0] >= (1.0 if vol is None else vol.max())

def includes_minvalue(trap, vol=None, eps=1e-2):
    return trap[0, 0] <= (eps if vol is None else vol.min() + eps)

def colorize_trapeze(t, color):
    res = np.zeros((t.shape[0], 5))
    res[:, 0]   = t[:, 0]
    res[:, 1:4] = color
    res[:, 4]   = t[:, 1]
    return res

def get_tf_pts_from_peaks(peaks, height_range=(0.1, 0.9), width_range=(0.02, 0.2), max_num_peaks=5):
    ''' Compute transfer function with trapezoids around given peaks
    Args:
        peaks (np.array of [intensity, persistence]): The histogram peaks
        height_range (tuple of floats): Range in which to draw trapezoid height (=opacity). Max range is (0, 1)
        width_range (tuple of floats): Range in which to draw trapezoid width around peak. Max range is (0, 1)
        max_num_peaks (int): Maximum number of peaks in the histogram. The number will be drawn as U(1, max_num_peaks)
    Returns:
        [ np.array [x, y] ]: List of TF primitives (List of coordinates [0,1]²) to be lerped
    '''
    num_peaks = np.random.randint(1, max_num_peaks)
    height_range_len = height_range[1] - height_range[0]
    width_range_len  = width_range[1] - width_range[0]
    color_gen = color_generator()
    def make_trapezoid(c, top_height, bot_width):
        bot_height = np.random.rand(1).item() * top_height
        top_width  = np.random.rand(1).item() * bot_width
        return np.stack([
          np.array([c - bot_width/2 -1e-2, 0]),    # left wall          ____________  __ top_height
          np.array([c - bot_width/2, bot_height]), # bottom left       / top_width  \
          np.array([c - top_width/2, top_height]), # top left        /__ bot_width __\__ bot_height
          np.array([c + top_width/2, top_height]), # top right      |                |
          np.array([c + bot_width/2, bot_height]), # bottom right   |   right wall ->|
          np.array([c + bot_width/2 +1e-2, 0])     # right wall     |<- left wall    |
        ])                                         #               |        c       |__ 0

    trapezes = [make_trapezoid(c, # Center of peak
        top_height= height_range_len * np.random.rand(1).item() + height_range[0],
        bot_width = width_range_len  * np.random.rand(1).item() + width_range[0]
        ) for c, p in peaks]
    result = []
    for t in trapezes:
        if overlaps_trapeze(t, result) or includes_maxvalue(t) or includes_minvalue(t): continue
        else: result.append(colorize_trapeze(t, next(color_gen)))
        if len(result) >= max_num_peaks: break
    res_arr = np.stack(result)
    np.random.shuffle(res_arr)
    res_arr = np.clip(res_arr[:num_peaks].reshape((-1, 5)), 0, 1)
    idx = np.argsort(res_arr[:, 0])
    return res_arr[idx]

def random_tf_from_vol(vol, max_num_peaks=5, height_range=(0.1, 0.9), width_range=(0.02, 0.2), bins=1000):
    if torch.is_tensor(vol): vol = vol.detach().cpu().float().numpy()
    peaks = get_histogram_peaks(vol, bins=bins)
    tf    = get_tf_pts_from_peaks(peaks, height_range=height_range, width_range=width_range, max_num_peaks=max_num_peaks)
    return torch.cat([torch.zeros(5)[None], torch.from_numpy(tf).float(), torch.tensor([[1,0,0,0,0.0]])])
