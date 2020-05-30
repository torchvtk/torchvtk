# Very much inspired by the excellent code from Philipp Henzler on his PlatonicGAN
# See https://github.com/henzler/platonicgan/blob/master/scripts/renderer
#%%
import torch
import torch.nn.functional as F
from torch import nn

#%%
class VolumeRaycaster(nn.Module):
    def __init__(self, density_factor=100, ray_samples=512, resolution=(640,480),
        device=torch.device('cpu'), dtype=torch.float32):
        ''' Initializes differentiable raycasting layer
        Args:
            density_factor (float): scales the overall density
            ray_samples (int): Number of samples along the rays
            resolution (int, (int, int)): Tuple describing width and height of the render. A single int produces a square image
            device (torch.Device): The device on which sample coordinates are stored.
            dtype (torch.dtype): Data type used for Raycasting. Choose either torch.float32 or torch.float16
            '''
        super().__init__()
        self.density_factor = density_factor
        self.ray_samples    = ray_samples
        self.dtype          = dtype
        self.device         = device
        if isinstance(resolution, tuple):
              self.w, self.h = resolution
        else: self.w, self.h = resolution, resolution

        Z = torch.linspace(-1, 1, ray_samples)
        W = torch.linspace(-1, 1, self.w)
        H = torch.linspace(-1, 1, self.h)
        self.samples = self.get_coord_grid(Z, H, W).to(device).to(dtype)

    def get_coord_grid(self, z, y, x):
        z, y, x = torch.meshgrid(z, y, x)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, vol):
        # vol = F.pad(vol, [1]*6)
        density = vol[:, [3]]
        color   = vol[:, :3]
        # Expand for all items in batch
        sample_coords = self.samples.expand(color.size(0), -1, -1, -1, -1)
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
        render = torch.sum(weight * color, dim=2) / (w_sum + 1e-8)
        alpha  = 1.0 - torch.prod(1 - density, dim=2)
        render = render * alpha
        # Concatenate to RGBA image
        return torch.cat([render, alpha], dim=1)
