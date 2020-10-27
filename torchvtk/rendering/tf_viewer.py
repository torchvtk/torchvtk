import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvtk.utils import tex_from_pts


def show(img, ax=plt, title=None, interpolation='nearest'):
    '''Displays a Tensor as image

    Args:
        img (torch.Tensor): Tensor containing the image. Shape (C, H, W)
        ax (pyplot.Axis, optional): Axis to show image in. Defaults to plt.
        title (str, optional): Title of the plot. Defaults to None.
        interpolation (str, optional): Interpolation mode for pyplot. Usually 'linear' or 'nearest'. Defaults to 'nearest'.
    '''
    npimg = img.squeeze().detach().float().cpu().numpy()
    if title is not None: ax.set_title(title)
    ax.tick_params(axis='both', which='both',
        bottom=False, right=False, left=False, labelbottom=False, labelleft=False)
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
    im = tf.detach().squeeze().unsqueeze(-2).expand(-1, 50, -1).contiguous().cpu()
    if   im.size(0) == 3: show(im, ax=ax, title=title)
    elif im.size(0) == 4:
        im[:3][:, im[3] == 0.0] = 0.0
        show(im[:3], ax=ax, title=title)
        ax.plot(range(im.size(2)), 50 - im[3, 0] * 50)

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
