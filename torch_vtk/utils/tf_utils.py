#%%
import torch
import torch.nn.functional as F
import numpy as np

from xml.dom import minidom
from torchinterp1d import Interp1d


def read_inviwo_tf(fn):
    xmldoc = minidom.parse(str(fn))
    def parse_point(point):
        pos  = float(point.getElementsByTagName('pos')[0].getAttribute('content'))
        _, rgba = zip(*point.getElementsByTagName('rgba')[0].attributes.items())
        return pos, list(map(float, rgba))
    points = sorted(map(parse_point, xmldoc.getElementsByTagName('Point')))
    flat_points = list(map(lambda t: [t[0]] + t[1], points))

    return torch.cat([torch.zeros(5)[None], torch.tensor(flat_points), torch.eye(5)[None, 0]], dim=0)

def tex_from_pts(tf_pts, resolution=4096):
    ''' Interpolates `tf_pts` to generate a TF texture of shape (N, C, resolution) with C determined by the TF '''
    if isinstance(tf_pts, np.ndarray):
        tf_pts = torch.from_numpy(tf_pts)
    if torch.is_tensor(tf_pts):
        return apply_tf_torch(torch.linspace(0.0, 1.0, resolution)[None,None], tf_pts)

def apply_tf_torch(x, tf_pts):
    ''' Applies the TF described by points `tf_pts` (N x [0,1]^C+1 with x pos and C channels) to `x`
    The operation always computes on torch.float32. The output is cast to `x.dtype`
    Args:
        x (torch.Tensor): The intensity values to apply the TF on. Assumed shape is (N, 1, ...) with batch size N
        tf_pts (torch.Tensor): Tensor of shape (N, (1+C)) containing N points consisting of x coordinate and mapped features (e.g. RGBO)
    Returns:
        torch.Tensor: Tensor with TF applied of shape (N, C, ...) with batch size N (same as `x`) and number of channels C (same as `tf_pts`)
    '''
    if isinstance(tf_pts, list): return torch.cat([apply_tf_torch(x, tf) for tf in tf_pts], dim=0) # If tf_pts is in a list, perform for each item in that list
    dev = x.device
    tf_pts = tf_pts.to(dev)
    npt, nc = tf_pts.shape
    x_shap = tuple(x.shape)
    x_out_shap = (x_shap[0], nc-1, *x_shap[2:])
    x_acc   = torch.empty(x_out_shap,        dtype=torch.float32, device=dev)
    pts_acc = torch.empty((npt * (nc-1), 2), dtype=torch.float32, device=dev)
    for i in range(1,nc):
        x_acc[:, i-1] = x + (i-1) # make intensity volume of shape (N, nc, W,H,D), with intensity values offset by 1 for each channel
        pts_acc[(i-1)*npt:i*npt] = tf_pts[:, [0,i]] + torch.Tensor([[i-1,0.0]]).to(dev) # offset TF values (xRGBO) similarly to get all channels aligned to intensity [0, nc-1]
    return Interp1d()(pts_acc[:,0].float(), pts_acc[:,1].float(), x_acc.float().view(-1)).reshape(x_out_shap).to(x.dtype) # Interp on flattened volume, reshape


# %%
