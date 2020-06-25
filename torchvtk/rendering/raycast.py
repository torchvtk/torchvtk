# Very much inspired by the excellent code from Philipp Henzler on his PlatonicGAN
# See https://github.com/henzler/platonicgan/blob/master/scripts/renderer
#%%
import math
import torch
import torch.nn.functional as F
from torch import nn

from torchvtk.utils.volume_utils import make_2d

__all__ = ['homogenize_mat', 'homogenize_vec', 'get_proj_mat', 'get_view_mat', 'get_random_pos', 'VolumeRaycaster']

def homogenize_mat(mat):
    ''' Adds a row (bottom) and column (right) to the matrix `mat` with all zeros and a 1 in the lower right corner.
    Is used to make the matrix work as transformation for homogeneous coordinates. '''
    assert torch.is_tensor(mat)
    ret = torch.eye(mat.size(-1)+1, dtype=mat.dtype, device=mat.device)
    if mat.ndim > 2:
        flat_mat = mat.view(-1, mat.size(-2), mat.size(-1))
        num_mats = flat_mat.size(0)
        ret = ret[None].expand(num_mats, -1, -1)
    else: flat_mat = mat
    ret[..., :mat.size(-2), :mat.size(-1)] = flat_mat
    return ret.reshape(*mat.shape[:-2], mat.size(-2)+1, mat.size(-1)+1)

def homogenize_vec(vec):
    ''' Adds an additional component to `vec` with value 1 to make it a homogeneous coordinate. '''
    assert torch.is_tensor(vec)
    ret = torch.eye(vec.size(-1)+1, dtype=vec.dtype, device=vec.device)[-1]
    if vec.ndim > 1:
        flat_vec = vec.view(-1, vec.size(-1))
        num_vecs = flat_vec.size(0)
        ret = ret[None].expand(num_vecs, -1)
    else: flat_vec = vec
    ret[..., :vec.size(-1)] = flat_vec
    return ret.reshape(*vec.shape[:-1], vec.size(-1)+1)


def get_proj_mat(fov, aspect, near=0.1, far=100, dtype=None, device=None):
    ''' Computes a projection matrix according to inputs
    Args:
        fov (float): Field of View in radians
        aspect (float): Aspect ratio width/height of the viewport
        near (float): Near plane distance
        far (float): Far plane distance
        dtype (torch.dtype): Torch type to cast matrix to
        device (torch.device): Device to put matrix on
    Returns:
        Perspective projection matrix as torch tensor of shape (4,4)
    '''
    q = 1 / math.tan(fov * 0.5)
    a = q / aspect
    # b = (far + near) / (near - far)
    # c = (2*far*near) / (near - far)
    b = -far / (far-near)
    c = - (far*near)/(far-near)

    return torch.tensor([[a, 0, 0, 0],
                         [0, q, 0, 0],
                         [0, 0, b,-1],
                         [0, 0, c, 0]]).to(dtype).to(device)

def get_view_mat(look_from, look_to=None, look_up=None, dtype=None):
    ''' Computes a view matrix based on camera parameters.
    Args:
        look_from (Tensor (BS, 3)): Batch of camera positions
        look_up (Tensor (BS, 3)): Batch of up vectors
        look_to (Tensor (BS, 3)): Batch of position vectors of the object of interest
    Returns:
        View matrix as torch Tensor of shape (BS, 3, 3). Uses device of `look_from`
    '''
    look_from = make_2d(look_from)
    bs, dev = look_from.size(0), look_from.device
    if look_up is None: look_up = F.normalize(torch.eye(3)[1].expand(bs, 3), dim=1).to(dev)
    else:               look_up = make_2d(look_up)
    if look_to is None: look_to = F.normalize(torch.zeros(3).expand( bs, 3), dim=1).to(dev)
    else:               look_to = make_2d(look_to)

    z = F.normalize(look_from - look_to, dim=1).expand(bs, 3)
    x = F.normalize(torch.cross(look_up, z), dim=1)
    y = torch.cross(z, x, dim=1)
    ret = torch.eye(4)[None].expand(bs, -1, -1)
    ret[..., :3, :3] = torch.stack([x, y, z], dim=1)
    ret[...,  3, :3] = torch.stack([torch.bmm(x.unsqueeze(-2), look_from.unsqueeze(-1)),
                                    torch.bmm(y.unsqueeze(-2), look_from.unsqueeze(-1)),
                                    torch.bmm(z.unsqueeze(-2), look_from.unsqueeze(-1))]).squeeze()
    return ret.to(dtype)

def get_rot_mat(look_from, old_look_from=None):
    if old_look_from is None:
        old_look_from = torch.zeros_like(look_from)
        old_look_from[..., 2] = 1.0
    look_from     = F.normalize(look_from, dim=1)
    old_look_from = F.normalize(old_look_from, dim=1)
    v = torch.cross(old_look_from, look_from, dim=1)
    s = torch.norm(v, dim=1)
    c = torch.bmm(old_look_from.unsqueeze(-2), look_from.unsqueeze(-1))
    vx = torch.tensor([[0,       -v[:,2], v[:,1]], # skew symmetric cross product matrix
                       [ v[:,2],    0,   -v[:,0]],
                       [-v[:,1],  v[:,0],    0  ]])
    return torch.eye(3) + vx + vx**2 * (1/(1+c))
def get_random_pos(bs=1, distance=(1,5)):
    ''' Computes a vector of random positions.
    Args:
        bs (int): Batch size, number of positions to generate
        distance (float, tuple of floats): Either a fixed distance or a range from which the distance is sampled uniformly
    Returns:
        List / Batch of random camera positions as torch Tensor of shape (BS, 3)
    '''
    if   isinstance(distance, (tuple, list)): # Draw random in between
        d = torch.rand(bs, 1) * (distance[1] - distance[0]) + distance[0]
    elif isinstance(distance, (int, float)):
        d = distance
    return F.normalize(torch.randn(bs, 3)) * d

#%%
class VolumeRaycaster(nn.Module):
    def __init__(self, density_factor=100, ray_samples=512, resolution=(640,480)):
        ''' Initializes differentiable raycasting layer
        Args:
            density_factor (float): scales the overall density
            ray_samples (int): Number of samples along the rays
            resolution (int, (int, int)): Tuple describing width and height of the render. A single int produces a square image
            '''
        super().__init__()
        self.density_factor = density_factor
        self.ray_samples    = ray_samples
        if isinstance(resolution, tuple):
              self.w, self.h = resolution
        else: self.w, self.h = resolution, resolution

        Z = torch.linspace(-1, 1, ray_samples)
        W = torch.linspace(-1, 1, self.w)
        H = torch.linspace(-1, 1, self.h)
        self.samples = self.get_coord_grid(Z, H, W)

    def get_coord_grid(self, z, y, x):
        z, y, x = torch.meshgrid(z, y, x)
        return torch.stack([x, y, z], dim=-1)

    def get_perspective_coord_grid(self, view_mat):
        bs = view_mat.size(0)
        corners = torch.tensor([
            [-1,-1,-1,    1],
            [-1,-1, 1,    1],
            [-1, 1,-1,    1],
            [-1, 1, 1,    1],
            [ 1,-1,-1,    1],
            [ 1,-1, 1,    1],
            [ 1, 1,-1,    1],
            [ 1, 1, 1,    1.0]]).expand(bs, -1, -1)
        c_vs = torch.bmm(corners, view_mat) # Corners in View space
        dist = torch.norm(c_vs[..., :3], dim=2)
        nears, fars = torch.clamp(dist.min(dim=1).values - 2, 0), dist.max(dim=1).values + 2
        c_h = F.normalize(c_vs[...,[0,2]].reshape(-1, 2), dim=1)
        c_v = F.normalize(c_vs[...,[1,2]].reshape(-1, 2), dim=1)
        vd  = F.normalize(torch.mean(c_vs, dim=1), dim=1).unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)

        alphas_h = torch.acos(torch.bmm(c_h.unsqueeze(-2), vd[...,[0,2]].unsqueeze(-1)).reshape(bs, 8))
        alphas_v = torch.acos(torch.bmm(c_v.unsqueeze(-2), vd[...,[1,2]].unsqueeze(-1)).reshape(bs, 8))
        fovs_h   = torch.max(alphas, dim=1).values
        fovs_v   = torch.max(alphas, dim=1).values
        ars  = fovs_h / fovs_v
        fovs = fovs_v + 0.05 # About 2.8deg larger than necessary to fit the volume in the frustum

        proj_mat = torch.stack([get_proj_mat(fov, ar, near, far) for fov, ar, near, far in zip(fovs, ars, nears, fars)])

        inv_view, inv_proj = torch.inverse(view_mat), torch.inverse(proj_mat)
        inv_tfm = torch.bmm(view_mat, proj_mat)
        z, y, x = torch.meshgrid(torch.linspace(-1, 1, self.ray_samples),
                                 torch.linspace(-1, 1, self.h),
                                 torch.linspace(-1, 1, self.w))
        samples = torch.stack([x,y,z, torch.ones_like(x)], dim=-1).expand(bs,-1,-1,-1,-1)
        samples_poses = torch.bmm(samples.reshape(bs, -1, 4), inv_tfm).reshape(bs, self.ray_samples, self.h, self.w, 4)

    def forward(self, vol, tfm=None, output_alpha=False):
        ''' Renders a volume (using given transforms) using raycasting.
        Args:
            vol (Tensor): Batch of volumes to render. Shape (BS, C, D, H, W)
            tfm (Tensor or function): Either a (BS, 4, 4) transformation matrix or a function
                that transforms sample coordinates of shape `ray_samples, height, width, 4`.
            output_alpha (bool): Whether to output RGBA instead of RGB. Default is False
        Returns:
            Batch of projected images of shape (BS, 3, H, W) with RGB (and optionally Alpha) channels
        '''
        density = vol[:, [3]]
        color   = vol[:, :3]
        bs = color.size(0)
        # Expand for all items in batch
        sample_coords = self.samples.expand(bs, -1, -1, -1, -1).to(vol.device).to(vol.dtype)
        if torch.is_tensor(tfm):
            tfm = tfm.to(vol.dtype).to(vol.device)
            n_ch = tfm.size(-1)
            if   sample_coords.size(-1) < n_ch: sample_coords = homogenize_vec(sample_coords)
            elif sample_coords.size(-1) > n_ch: sample_coords = sample_coords[..., :n_ch]
            sample_coords = (torch.bmm(tfm, sample_coords.reshape(bs, -1, n_ch).permute(0,2,1))
                # .permute(0, 2, 1) # Permute back
                .view(bs, self.ray_samples, self.h, self.w, n_ch)
                .contiguous())
            sample_coords[..., :3] /= sample_coords[..., [3]] # divide by homogeneous
            sample_coords = sample_coords[..., :3]
            self.sample_coords = sample_coords.cpu()
        elif tfm is None: pass
        else: sample_coords = tfm(sample_coords)
        # Compute opacity and transmission along rays
        density = self.density_factor * density / self.ray_samples
        density = F.grid_sample(density, sample_coords)
        transmission = torch.cumprod(1.0 - density, dim=2)
        # Get sample weighting
        weight = density * transmission
        w_sum  = torch.sum(weight, dim=2)
        # Sample colors
        color = F.grid_sample(color, sample_coords)
        # Composite alpha and colors
        render = torch.sum(weight * color, dim=2) / (w_sum + 1e-6)
        alpha  = 1.0 - torch.prod(1 - density, dim=2)
        render = render * alpha
        # Concatenate to RGBA image
        if output_alpha: return torch.cat([render, alpha], dim=1)
        else:            return render
