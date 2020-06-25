from setuptools import setup, find_packages
setup(
    name="torchvtk",
    description="Efficient data loading and visualization for volumes in PyTorch",
    version="0.1",
    packages=['torchvtk'],
    install_requires=[
        'torch', 'torchvision', 'numpy', 'matplotlib'
    ],
    dependency_links=[
        'https://github.com/aliutkus/torchsearchsorted.git@master#egg=torchsearchsorted',
        'https://github.com/aliutkus/torchinterp1d.git@master#egg=torchinterp1d'
    ]
)
