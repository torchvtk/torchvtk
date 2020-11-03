from functools import partial

import numpy as np
import torch
import colorsys

from torchvtk.rendering import VolumeRaycaster, plot_comp_render_tf
from torchvtk.utils import make_5d, tex_from_pts

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


def distinguishable_color_generator():
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

def random_color_generator():
    ''' Generates random colors '''
    while True:
        h, s, l = np.random.rand(), 0.2 + np.random.rand() * 0.8, 0.35 + np.random.rand() * 0.3
        yield np.array([float(255*i) for i in colorsys.hls_to_rgb(h,l,s)], dtype=np.float32) / 255.0

def fixed_color_generator(color=(180, 40, 40.0)):
    while True: yield np.array(color).astype(np.float32) / 255.0

def get_histogram_peaks(data, bins=1024, skip_outlier=True):
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

def flatten_clip_sort_peaks(peaks):
    if len(peaks) == 0:
        peaks = np.zeros((1,5))
    arr = np.clip(np.stack(peaks).reshape((-1, 5)), 0, 1)
    idx = np.argsort(arr[:, 0])
    return arr[idx]

def colorize_trapeze(t, color):
    res = np.zeros((t.shape[0], 5))
    res[:, 0]   = t[:, 0]
    res[:, 1:4] = color
    res[:, 4]   = t[:, 1]
    return res

def get_tf_pts_from_peaks(peaks, colors='random', height_range=(0.1, 0.9), width_range=(0.02, 0.2), peak_center_noise_std=0.05, max_num_peaks=5, peak_valid_fn=None):
    ''' Compute transfer function with non-overlapping trapezoids around given peaks

    Args:
        peaks (np.array of [intensity, persistence]): The histogram peaks
        colors (str): Either "distinguishable", "random" or "fixed"
        height_range (tuple of floats): Range in which to draw trapezoid height (=opacity). Max range is (0, 1)
        width_range (tuple of floats): Range in which to draw trapezoid width around peak. Max range is (0, 1)
        peak_center_noise_std (float): Standard deviation of the Gaussian noise applied to peak centers, to shift those randomly.
        max_num_peaks (int): Maximum number of peaks in the histogram. The number will be drawn as U(1, max_num_peaks)
        peak_valid_fn (func): Function that gets the old TF without a new peak and the TF with the new peak and decides wether to add the peak (return True) or not (return False).

    Returns:
        [ np.array [x, y] ]: List of TF primitives (List of coordinates [0,1]²) to be lerped
    '''
    if peak_valid_fn is None: peak_valid_fn = lambda a, b: True
    n_peaks = np.random.randint(1, max_num_peaks+1)
    height_range_len = height_range[1] - height_range[0]
    width_range_len  = width_range[1] - width_range[0]
    if   colors == 'distinguishable': color_gen = distinguishable_color_generator()
    elif colors == 'random':          color_gen = random_color_generator()
    elif colors == 'fixed':           color_gen = fixed_color_generator()
    else: raise Exception(f'Invalid colors argument ({colors}). Use either "distinguishable" or "random".')
    def make_trapezoid(c, top_height, bot_width):
        c += np.random.randn() * peak_center_noise_std
        bot_width = bot_width * c + 1e-2     # allow for wider peaks in higher density
        int_contrib = np.clip(c * (1/0.6), 0, 1) # higher opacity on higher density (usually bones, which are often occluded)
        top_height = (int_contrib + top_height)  / 2.0 # allow for mostly low peaks on skin, higher peaks on bones
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

    if peaks is None:
        peaks = np.random.rand(100, 2)
    trapezes = [make_trapezoid(c, # Center of peak
        top_height= height_range_len * np.random.rand(1).item() + height_range[0],
        bot_width = width_range_len  * np.random.rand(1).item() + width_range[0]
        ) for c, p in peaks]
    result = []
    np.random.shuffle(trapezes)
    fail_count = 0
    for t in trapezes:
        if overlaps_trapeze(t, result) or includes_maxvalue(t) or includes_minvalue(t):
            continue
        else:
            trap = colorize_trapeze(t, next(color_gen))
            if peak_valid_fn(
              tf_pts_border(flatten_clip_sort_peaks(result)),
              tf_pts_border(flatten_clip_sort_peaks(result + [trap]))):
                fail_count = 0  # reset fail count if peak gets added
                result.append(trap)
            else: fail_count += 1 # failed in that the new TF does produce a too similar image
        if len(result) >= n_peaks or fail_count > 5: break # max 5 render tries
    return flatten_clip_sort_peaks(result)

def tf_pts_border(tf_pts):
    if isinstance(tf_pts, np.ndarray):
        tf_pts = torch.from_numpy(tf_pts)
    l = torch.zeros(1, tf_pts.size(-1))
    r = torch.eye(tf_pts.size(-1))[None, 0]
    return torch.cat([l, tf_pts, r], dim=0)

def random_tf_from_vol(vol, colors='random', max_num_peaks=5, height_range=(0.1, 0.7), width_range=(0.02, 0.3), peak_center_noise_std=0.05, bins=1024, valid_fn=None, use_hist=True):
    if torch.is_tensor(vol): vol = vol.detach().cpu().float().numpy()
    peaks = get_histogram_peaks(vol, bins=bins) if use_hist else None
    tf    = get_tf_pts_from_peaks(peaks, colors=colors, height_range=height_range, width_range=width_range, max_num_peaks=max_num_peaks, peak_center_noise_std=peak_center_noise_std, peak_valid_fn=valid_fn)
    return tf_pts_border(tf)

class TFGenerator():
    def __init__(self, mode='random_peaks', colors='random', peakgen_kwargs={}, raycast_kwargs={}):
        ''' Initializes TF Generator

        Args:
            mode (str, optional): Either 'random_peaks' for random non-overlapping trapezoids or 'verified_peaks' for additionally ensuring that each peak can be seen from a given. Defaults to 'random_peaks'.
            colors (str, optional): Either 'random', 'distinguishable' or 'fixed'. Defaults to 'random'.
            peakgen_kwargs (dict, optional): Keyword args for the peak generation through `random_tf_from_vol`. Defaults to {}.
            raycast_kwargs (dict, optional): Keyword args for the Volume Raycaster used for peak validation, see `torchvtk.rendering.VolumeRaycaster`. Defaults to {}.
        '''
        self.mode = mode
        self.peakgen_kwargs = {
            'max_num_peaks': 5,
            'height_range': (0.02, 0.9),
            'width_range': (0.02, 0.3),
            'peak_center_noise_std': 0.05,
            'max_num_peaks': 5,
            'bins': 1024,
            'colors': colors,
            'use_hist': True
        }
        self.peakgen_kwargs.update(peakgen_kwargs)
        self.raycast_kwargs = {
            'density_factor': 100.0,
            'ray_samples': 128,
            'resolution': (128, 128)
        }
        self.raycast_kwargs.update(raycast_kwargs)
        if self.mode == 'verified_peaks':
            self.raycaster = VolumeRaycaster(**self.raycast_kwargs)
        self.figs = []

    def validate_peak(self, tf_before, tf_after, vol, view_mat, l1_thresh=1e-2):
        if isinstance(tf_before, np.ndarray):
            tf_before, tf_after = torch.from_numpy(tf_before), torch.from_numpy(tf_after)
        im_before = self.last_im
        im_after  = self.raycaster(make_5d(vol), tf=[tf_after],  view_mat=view_mat)

        l1 = (im_before - im_after).abs()
        if l1.median() + l1.mean() > l1_thresh:
            self.last_im = im_after
            return True
        else:
            return False

    def generate(self, vol, view_mat=None):
        if view_mat is not None and self.mode == 'verified_peaks':
            self.peakgen_kwargs['valid_fn'] = partial(self.validate_peak,
                vol=vol, view_mat=view_mat)
            self.last_im = torch.zeros(1,3,*self.raycast_kwargs['resolution'], dtype=vol.dtype)
        else:
            self.peakgen_kwargs['valid_fn'] = None
        return random_tf_from_vol(vol, **self.peakgen_kwargs).to(torch.float32)
