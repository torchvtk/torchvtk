#%%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from torchvtk.rendering.raycast import *
from torchvtk.utils.tf_utils    import apply_tf_torch, read_inviwo_tf
from torchvtk.datasets          import TorchDataset

# TODO: Add tests whether results with different dtypes (maybe devices) are similar.
# TODO: Also maybe save reference image to reproduce in tests



#%%

if __name__ == '__main__':
# %%

    vr = VolumeRaycaster(ray_samples=512, density_factor=128, resolution=(224,224))
    print(vr.samples[0,0,0])
    print(vr.samples[0,0,-1])
    print(vr.samples[0,-1,0])
    print(vr.samples[0,-1,-1])
    print(vr.samples[-1,0,0])
    print(vr.samples[-1,0,-1])
    print(vr.samples[-1,-1,0])
    print(vr.samples[-1,-1,-1])

# %%
    dtype  = torch.float32
    device = torch.device('cpu')
    cq500_path = Path('/run/media/dome/Data/data/torchvtk/CQ500')

    a = torch.load(f'/run/media/dome/Data/data/deep-tf/CQ500-CT-0_0017.pt')
    vol_path = cq500_path/(a['vol_name'].replace('-', '_') + '.pt')
    data = torch.load(vol_path)
    vol = data['vol'][None, None].to(dtype)
    vol = F.interpolate(vol, size=(256,256,256), mode='trilinear')

    # %%
    tf = a['tf']
    vm = a['view_mat']
    vm[:,2] *= -1 # flip z basis
    v = apply_tf_torch(vol, [tf])

    im = vr.forward(v, view_mat=vm[None])
    nzp = (im.squeeze() > 1e-2).sum(dim=[0,2]).bool().tolist().index(True)
    nzt = (a['render']  > 1e-2).sum(dim=[0,2]).bool().tolist().index(True)
    print(f"First nonzero row pred: {nzp}/{im.size(2)} ({nzp / im.size(2)})")
    print(f"First nonzero row targ: {nzt}/{a['render'].size(2)} ({nzt / a['render'].size(2)})")

    im = im.squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(im)
    axs[1].imshow(a['render'].permute(1,2,0).numpy())
    fig.show()





 # %%


# %%
