#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from torch_volume_dataloaders.rendering.raycast import VolumeRaycaster

# TODO: Add tests whether results with different dtypes (maybe devices) are similar.
# TODO: Also maybe save reference image to reproduce in tests
#%%
if __name__ == '__main__':
#%%
    dtype  = torch.float32
    device = torch.device('cpu')
    v = np.load('data/boron.npy')
    v = torch.from_numpy(v.astype(np.float32)) / 2**16
    v = v[None,None].expand(1, 4, -1, -1, -1)
    vr = VolumeRaycaster(device=device, dtype=dtype, ray_samples=256)
    v = v.to(dtype).to(device)
    t0 = time.time()
    im = vr.forward(v).squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)
    print('Time elapsed:', time.time() - t0)
    plt.imshow(im)


# %%
