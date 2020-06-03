"""
From https://github.com/henzler/platonicgan/blob/master/scripts/renderer/transform.py

"""


import torch
import numpy as np
import scipy.ndimage
import math


class RotationHelper():
    def __init__(self, device):
        super().__init__()
        self.device = device

    def rotate(self, volume, rotation_matrix):

        batch_size = volume.shape[0]
        size = volume.shape[2]

        indices = self.get_graphics_grid_coords_3d(size, size, size, coord_dim=0)
        indices = indices.expand(batch_size, 3, size, size, size)
        indices = indices.to(self.device)

        indices_rotated = torch.bmm(rotation_matrix, indices.view(batch_size, 3, -1)).view(batch_size, 3, size, size, size)

        return self.resample(volume, indices_rotated)

    def rotate_random(self, volume):

        batch_size = volume.shape[0]
        rotation_matrix = self.get_rotation_matrix_random(batch_size)

        return self.rotate(volume, rotation_matrix)

    def rotate_by_vector(self, volume, vector):

        rotation_matrix = self.get_view_matrix_by_vector(vector)

        return self.rotate(volume, rotation_matrix)

    def get_vector_random(self, batch_size):

        theta = torch.empty((batch_size), dtype=torch.float).uniform_(0, 2 * np.pi).to(self.device)
        u = torch.empty((batch_size), dtype=torch.float).uniform_(-1.0, 1.0).to(self.device)

        x = torch.sqrt(1 - u * u) * torch.cos(theta)
        y = torch.sqrt(1 - u * u) * torch.sin(theta)
        z = u

        vector = torch.stack([x, y, z], dim=1)

        return vector

    def get_rotation_matrix_random(self, batch_size):

        vector = self.get_vector_random(batch_size)
        random_view_matrix = self.get_view_matrix_by_vector(vector)
        return random_view_matrix

    def get_view_matrix_by_vector(self, eye, up_vector=None, target=None):

        bs = eye.size(0)

        if up_vector is None:
            up_vector = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0]).expand(bs, 3), dim=1).to(self.device)

        if target is None:
            target = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 0.0]).expand(bs, 3), dim=1).to(self.device)

        z_axis = torch.nn.functional.normalize(eye - target, dim=1).expand(bs, 3)
        x_axis = torch.nn.functional.normalize(torch.cross(up_vector, z_axis), dim=1)
        y_axis = torch.cross(z_axis, x_axis, dim=1)
        view_matrix = torch.stack([x_axis, y_axis, z_axis], dim=2)

        return view_matrix.to(self.device)

    def resample(self, volume, indices_rotated):

        if volume.is_cuda:
            indices_rotated = indices_rotated.to('cuda')

        indices_rotated = indices_rotated.permute(0, 2, 3, 4, 1)

        # transform coordinate system
        # flip y and z
        # grid sample expects y- to be up and z- to be front
        indices_rotated[..., 1] = -indices_rotated[..., 1]
        indices_rotated[..., 2] = -indices_rotated[..., 2]
        volume = torch.nn.functional.grid_sample(volume, indices_rotated, mode='bilinear')

        return volume

    def get_graphics_grid_coords_3d(self, z_size, y_size, x_size, coord_dim=-1):
        steps_x = torch.linspace(-1.0, 1.0, x_size)
        steps_y = torch.linspace(1.0, -1.0, y_size)
        steps_z = torch.linspace(1.0, -1.0, z_size)
        z, y, x = torch.meshgrid(steps_z, steps_y, steps_x)
        coords = torch.stack([x, y, z], dim=coord_dim)
        return coords