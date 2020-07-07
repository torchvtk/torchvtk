import os

import torch
from batchviewer import view_batch

from torchvtk.augmentation.DictTransform import BlurDictTransform, NoiseDictTransform, CroppingTransform


def main():
    # import test image
    file = load_file()

    # double
    file["vol"] = torch.stack([file["vol"], file["vol"]], dim=0)
    view_batch(file["vol"][0, ...], width=512, height=512)

    # Test Noise Transform.
    tfms = NoiseDictTransform(noise_variance=0.01, device="cpu", apply_on=["vol"])
    tmp = tfms(file)
    view_batch(tmp["vol"].squeeze(), width=512, height=512)
    del tfms, file

    # Test Noise Transform GPU.
    file = load_file()

    tfms = NoiseDictTransform(device="cuda", apply_on=["vol"], noise_variance=0.01)
    tmp = tfms(file)
    view_batch(tmp["vol"].squeeze(), width=512, height=512)
    del tfms, file

    # test for gaussian blur cpu
    file = load_file()
    # double
    file["vol"] = torch.stack([file["vol"], file["vol"]], dim=0)
    tfms = BlurDictTransform(apply_on=["vol"], device="cpu", channels=1, kernel_size=(3, 3, 3), sigma=1)
    tmp = tfms(file)
    view_batch(tmp["vol"].squeeze(), width=512, height=512)
    del tfms, file

    # test for gaussian blur gpu
    file = load_file()
    tfms = BlurDictTransform(apply_on=["vol"], device="cuda", channels=1, kernel_size=(3, 3, 3), sigma=1)
    blur_gpu = tfms(file)
    view_batch(blur_gpu["vol"].squeeze(), width=512, height=512)
    file["vol"] = file["vol"].to("cpu")
    del tfms, file

    # Cropping CPU.
    file = load_file()
    tfms = CroppingTransform(device="cpu", apply_on=["vol", "mask"], dtype=torch.float32)
    tmp = tfms(file)
    view_batch(tmp["vol"].squeeze(), tmp["mask"].squeeze(), width=512, height=512)
    del tfms, file

    # Cropping CPU.
    file = load_file()
    # double
    file["vol"] = torch.stack([file["vol"], file["vol"]], dim=0)
    tfms = CroppingTransform(device="cuda", apply_on=["vol"], dtype=torch.float32)
    tmp = tfms(file)
    view_batch(tmp["vol"].squeeze(), width=512, height=512)


def load_file():
    file_path = "data/test_ct_images.pt"
    file = torch.load(file_path)
    return file


if __name__ == '__main__':
    main()

