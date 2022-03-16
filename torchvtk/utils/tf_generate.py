# %%
from functools import partial
import logging

import numpy as np
import torch
import colorsys

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
    np.random.shuffle(colors)
    for color in colors:
        yield color

def random_color_generator():
    ''' Generates random colors '''
    while True:
        h, s, l = np.random.rand(), 0.2 + np.random.rand() * 0.8, 0.35 + np.random.rand() * 0.3
        yield np.array([float(255*i) for i in colorsys.hls_to_rgb(h,l,s)], dtype=np.float32) / 255.0

def fixed_color_generator(color=(180, 170, 170.0)):
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

def make_trapezoid(c, top_height, bot_width, fixed_shape=False):
    # bot_width = bot_width * c + 1e-2     # allow for wider peaks in higher density
    # int_contrib = np.clip(c * (1/0.6), 0, 1) # higher opacity on higher density (usually bones, which are often occluded)
    # top_height = (int_contrib + top_height)  / 2.0 # allow for mostly low peaks on skin, higher peaks on bones
    if fixed_shape:
        bot_height = top_height
        top_width  = bot_width
    else:
        bot_height = np.random.rand(1).item() * top_height
        top_width  = np.random.rand(1).item() * bot_width
    return np.stack([
      np.array([c - bot_width/2 -2e-2, 0]),    # left wall          ____________  __ top_height
      np.array([c - bot_width/2, bot_height]), # bottom left       / top_width  \
      np.array([c - top_width/2, top_height]), # top left        /__ bot_width __\__ bot_height
      np.array([c + top_width/2, top_height]), # top right      |                |
      np.array([c + bot_width/2, bot_height]), # bottom right   |   right wall ->|
      np.array([c + bot_width/2 +2e-2, 0])     # right wall     |<- left wall    |
    ])

def get_tf_pts_from_peaks(peaks, colors='random', height_range=(0.1, 0.9), width_range=(0.02, 0.2), peak_center_noise_std=0.05, max_num_peaks=5, peak_valid_fn=None, fixed_shape=False):
    ''' Compute transfer function with non-overlapping trapezoids around given peaks

    Args:
        peaks (np.array of [intensity, persistence]): The histogram peaks
        colors (str): Either "distinguishable", "random" or "fixed"
        height_range (tuple of floats): Range in which to draw trapezoid height (=opacity). Max range is (0, 1)
        width_range (tuple of floats): Range in which to draw trapezoid width around peak. Max range is (0, 1)
        peak_center_noise_std (float): Standard deviation of the Gaussian noise applied to peak centers, to shift those randomly.
        max_num_peaks (int): Maximum number of peaks in the histogram. The number will be drawn as U(1, max_num_peaks)
        peak_valid_fn (func): Function that gets the old TF without a new peak and the TF with the new peak and decides wether to add the peak (return True) or not (return False).
        fixed_shape (bool): If True produces a classic ramp peak, if False it has random double ramps

    Returns:
        [ np.array [x, y] ]: List of TF primitives (List of coordinates [0,1]²) to be lerped
    '''
    if peak_valid_fn is None: peak_valid_fn = lambda a, b: True
    if max_num_peaks is None:
        n_peaks = len(peaks)
    elif isinstance(max_num_peaks, (tuple, list)) and len(max_num_peaks) == 2:
        n_peaks = np.random.randint(max_num_peaks[0], max_num_peaks[1] + 1)
    else:
        n_peaks = np.random.randint(1, max_num_peaks+1)
    height_range_len = height_range[1] - height_range[0]
    width_range_len  = width_range[1] - width_range[0]
    if   colors == 'distinguishable': color_gen = distinguishable_color_generator()
    elif colors == 'random':          color_gen = random_color_generator()
    elif colors == 'fixed':           color_gen = fixed_color_generator()
    else: raise Exception(f'Invalid colors argument ({colors}). Use either "distinguishable" or "random".')                                        #               |        c       |__ 0

    if peaks is None:
        peaks = np.random.rand(100, 2)
        peaks = np.stack([np.linspace(0.05, 0.75, 15)]*2, axis=1)
    trapezes = [make_trapezoid(c     + np.random.randn() * peak_center_noise_std, # Center of peak
        top_height= height_range_len * np.random.rand(1).item() + height_range[0],
        bot_width = width_range_len  * np.random.rand(1).item() + width_range[0],
        fixed_shape=fixed_shape
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

def create_peaky_tf(peaks, widths, default_color=(0.7, 0.66, 0.66), default_height=0.99, warn_overlap=True):
    ''' Creates a peaky tf with given peak centers, widths and optional rgb, o
    Beware: The output of this function is undefined for overlapping trapezes! A warning will be printed.

    Args:
        peaks (array): Array of shape (N) only peak centers / (N, 2) centers and opacity / (N, 4) centers and rgb / (N, 5) centers, opacity and rgb.
        widths (array): Array of shape (N), same length as peaks.
        default_color (array, optional): RGB value as array. Defaults to (0.7, 0.66, 0.66).
        default_height (float, optional): Default opacity of none is given in peaks. Defaults to 0.99.
        warn_overlap (bool, optional): Prints a warning if the resulting Transfer Function has overlapping trapezes. Defaults to True.

    Returns:
        Array: Point-based Transfer Function (N, 5) with the given peaks
    '''
    trapezes = []
    for p, w in zip(peaks, widths):
        if not hasattr(p, '__len__'): c, o, rgb = p, default_height, default_color
        elif len(p) == 2:             c, o, rgb = p[0], p[1], default_color
        elif len(p) == 4:             c, o, rgb = p[0], default_height, p[1:]
        elif len(p) == 5:             c, o, rgb = p[0], p[1], p[2:]
        else: raise Exception(f'Invalid input for peaks: list of {p}. See docstring of create_peaky_tf()')
        if warn_overlap and overlaps_trapeze(make_trapezoid(c, o, w, fixed_shape=True), trapezes):
            logging.warning(f'create_peaky_tf() has overlapping trapezes. First overlapping trapeze in the sequence: (center={c}, width={w}, index={len(trapezes)})')
        trapezes.append(colorize_trapeze(make_trapezoid(c, o, w, fixed_shape=True), rgb))

    return tf_pts_border(flatten_clip_sort_peaks(trapezes))

def create_cos_tf(phases, amplitudes, frequencies=range):
    n = len(phases)
    if not torch.is_tensor(phases): phases = torch.Tensor(phases)
    if not torch.is_tensor(amplitudes): amplitudes = torch.Tensor(amplitudes)
    if hasattr(frequencies, __call__):
        freqs = torch.Tensor(frequencies(n))
    else:
        assert len(frequencies) == n
    def tf(x):
        x.expand(*([-1]*x.ndim), n)
        torch.cos(freqs * (x + phases)) * amplitudes


def tries():
    n = 20
    amps = torch.rand(n)
    #freqs = torch.cat([torch.arange(2, 2+n//2), torch.arange(n//2, n)**1.4]).round()
    freqs = torch.arange(2, n+2)**1.2
    phases = torch.rand(n) * pi
    x = torch.linspace(0,1,100).unsqueeze(-1).expand(-1, n)
    plt.ylim((0,1))
    pts = (torch.cos(pi * freqs * (x + phases)) * amps).sum(-1) / (amps.sum() *0.5) + 0.5
    plt.title(f'{freqs.round().tolist()}')
    plt.stackplot(torch.linspace(0,1,100), pts)

def tf_pts_border(tf_pts):
    if isinstance(tf_pts, np.ndarray):
        tf_pts = torch.from_numpy(tf_pts)
    l = torch.zeros(1, tf_pts.size(-1))
    r = torch.eye(tf_pts.size(-1))[None, 0]
    return torch.cat([l, tf_pts, r], dim=0)

def random_tf_from_vol(vol, colors='random', max_num_peaks=5, height_range=(0.1, 0.7), width_range=(0.02, 0.3), peak_center_noise_std=0.05, bins=1024, valid_fn=None, use_hist=True, fixed_shape=False, override_peaks=None):
    if torch.is_tensor(vol): vol = vol.detach().cpu().float().numpy()
    if override_peaks is not None:
        if isinstance(override_peaks, np.ndarray):
            peaks = override_peaks
        elif torch.is_tensor(override_peaks):
            peaks = override_peaks.numpy()
        else:
            peaks = np.stack([np.linspace(1/20, 1-1/20, 19)]*2, axis=1)
    else:
        peaks = get_histogram_peaks(vol, bins=bins) if use_hist and vol is not None else None
    tf    = get_tf_pts_from_peaks(peaks, colors=colors, height_range=height_range, width_range=width_range, max_num_peaks=max_num_peaks, peak_center_noise_std=peak_center_noise_std, peak_valid_fn=valid_fn, fixed_shape=fixed_shape)
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
            'max_num_peaks': 4,
            'height_range': (0.02, 0.95),
            'width_range': (0.005, 0.1),
            'peak_center_noise_std': 0.05,
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
            self.raycaster = torchvtk.rendering.VolumeRaycaster(**self.raycast_kwargs)
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

    def generate(self, vol=None, view_mat=None):
        if view_mat is not None and self.mode == 'verified_peaks':
            self.peakgen_kwargs['valid_fn'] = partial(self.validate_peak,
                vol=vol, view_mat=view_mat)
            self.last_im = torch.zeros(1,3,*self.raycast_kwargs['resolution'], dtype=vol.dtype)
        else:
            self.peakgen_kwargs['valid_fn'] = None
        return random_tf_from_vol(vol, **self.peakgen_kwargs).to(torch.float32)


# %%
