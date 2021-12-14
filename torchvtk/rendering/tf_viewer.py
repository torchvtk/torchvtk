import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvtk.utils import tex_from_pts


def show(img, ax=plt, title=None, interpolation='nearest', xticks=False):
    '''Displays a Tensor as image

    Args:
        img (torch.Tensor): Tensor containing the image. Shape (C, H, W)
        ax (pyplot.Axis, optional): Axis to show image in. Defaults to plt.
        title (str, optional): Title of the plot. Defaults to None.
        interpolation (str, optional): Interpolation mode for pyplot. Usually 'linear' or 'nearest'. Defaults to 'nearest'.
        xticks (bool): Wether to show xticks. Defaults to False.
    '''
    npimg = img.detach().float().cpu().numpy()
    if title is not None: ax.set_title(title)
    ax.tick_params(axis='y' if xticks else 'both', which='both',
        bottom=False, right=False, left=False, labelbottom=False, labelleft=False)
    if xticks:
        ax.set_xticks(np.linspace(0, 1, 11) * (img.size(-1)-1))
        ax.set_xticklabels(list(map(lambda n: f'{n:0.1f}', np.linspace(0, 1, 11))))
    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation=interpolation)

def show_tf(tf, ax=plt, title=None):
    '''Actually creates a TF figure

    Args:
        tf (torch.Tensor): The TF, either Texture or Points
        ax (pyplot.Axis, optional): Axis to plot to. Defaults to plt.
        title (str, optional): Title of the plot. Defaults to None.
    '''
    if tf.ndim == 2 and tf.size(-1) <= 5:
        tf = tex_from_pts(tf, 256)
    h = tf.size(-1) // 5
    im = tf.detach().unsqueeze(-2).expand(-1, h, -1).contiguous().cpu()
    if im.dtype == torch.float16: im = im.float()
    if   im.size(0) == 1:
        imm = torch.cat([torch.zeros_like(im)]*3, dim=0)
        show(imm, ax=ax, title=title, xticks=True)
        ax.set_ylim((h, 0))
        ax.plot(range(im.size(-1)), h - im[0,0] * h)
    if   im.size(0) == 3: show(im, ax=ax, title=title)
    elif im.size(0) == 4:
        im[:3][:, im[3] < 1e-3] = 0.0
        show(im[:3], ax=ax, title=title, xticks=True)
        ax.set_ylim((h, 0))
        ax.plot(range(im.size(2)), h - im[3, 0] * h)

def plot_render_tf(render, tf, title=''):
    '''Plots a Rendering and Transfer Function Nicely atop of each other

    Args:
        render (torch.Tensor): Rendering in Tensor of shape (C, H, W)
        tf (torch.Tensor): Transfer Function, either as Texture (C, W) or a list of Points (N, C+1)
        title (str, optional): Name for this render/TF pair to be displayed as title. Defaults to ''.

    Returns:
        pyplot.Figure: pyplot figure of both Rendering and Transfer Function
    '''
    gs = {
        'width_ratios': [5],
        'height_ratios': [5, 1]
    }
    fig, axs = plt.subplots(2,1, gridspec_kw=gs, figsize=(5,6))
    show(render, axs[0], f'{title} Rendering')
    show_tf(tf,  axs[1], f'{title} Transfer Function')
    return fig

def plot_renders(renders):
    '''Plots multiple renders

    Args:
        renders (Iterable, Tensor): list or stacked renderings
    '''
    gs = {
        'width_ratios': [1]*len(renders),
        'height_ratios': [1]
    }
    fig, axs = plt.subplots(1, len(renders), gridspec_kw=gs, figsize=(len(renders) * 5, 5))
    for ren, ax in zip(renders, axs): show(ren, ax)
    return fig

def plot_render_2tf(render, tf1, tf2, title=''):
    gs = {
        'width_ratios': [5],
        'height_ratios': [5, 1, 1]
    }
    fig, axs = plt.subplots(3,1, gridspec_kw=gs, figsize=(5,7))
    show(render, axs[0], f'{title} Rendering / TF1 / TF2')
    show_tf(tf1, axs[1])
    show_tf(tf2, axs[2])
    return fig

def plot_tf(tf, title=''):
    '''Plots a Transfer Function as texture

    Args:
        tf (torch.Tensor): The transfer function. Either as Texture (C, W) or list of Points (N, C+1)
        title (str, optional): Name of the TF, to be used as Title. Defaults to ''.

    Returns:
        pyplot.Figure: pyplot figure of the Transfer Function Texture
    '''
    fig, ax = plt.subplots(1,1)
    show_tf(tf, ax, title)
    return fig

def plot_comp_render_tf(pairs):
    ''' Plots a comparison of render/TF pairs

    Args:
        pairs (List of Tuples): List of Tuples (Render, Transfer Function [, Name])

    Returns:
        pyplot.Figure: Pyplot figure with Renders/TFs arranged
    '''
    n = len(pairs)
    gs = {
        'width_ratios': [5]*n,
        'height_ratios': [5, 1]
    }
    fig, axs = plt.subplots(2, n, gridspec_kw=gs, figsize=(5*n, 6))
    for i, pair in enumerate(pairs):
        if len(pair) == 3:
            render, tf, title = pair
        else:
            render, tf = pair
            title = ''
        show(render, axs[0, i], f'{title} Rendering')
        show_tf(tf,  axs[1, i], f'{title} Transfer Function')
    return fig

def plot_9comp_render_tf(pairs):
    ''' Plots a comparison of render/TF pairs

    Args:
        pairs (List of Tuples): List of Tuples (Render, Transfer Function [, Name])

    Returns:
        pyplot.Figure: Pyplot figure with Renders/TFs arranged
    '''
    n = len(pairs)
    gs = {
        'width_ratios': [5,5,5],
        'height_ratios': [5,1,5,1,5,1]
    }
    fig, axs = plt.subplots(6, 3, gridspec_kw=gs, figsize=(15, 18))
    for i, pair in enumerate(pairs):
        if len(pair) == 3:
            render, tf, title = pair
        else:
            render, tf = pair
            title = ''
        show(render, axs[2*(i // 3),   i % 3], f'{title} Rendering')
        show_tf(tf,  axs[2*(i // 3)+1, i % 3], f'{title} Transfer Function')
    return fig


def plot_tfs(tfs, titles=None):
    if titles is None: titles = ['']*len(tfs)
    fig, axs = plt.subplots(len(tfs),1)
    for tf, ax, t in zip(tfs, axs, titles):
        show_tf(tf, ax, title=t)
    return fig
