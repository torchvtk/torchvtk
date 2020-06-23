from setuptools import setup, find_packages
setup(
    name="torch-vtk",
    description="Efficient data loading and visualization for volumes in PyTorch",
    version="0.1",
    packages=['torch_vtk'],
    install_requires=[
        'torch', 'torchvision', 'numpy', 'matplotlib'
    ],
    dependency_links=[
        'https://github.com/aliutkus/torchsearchsorted.git@master#egg=torchsearchsorted',
        'https://github.com/aliutkus/torchinterp1d.git@master#egg=torchinterp1d'
    ]
)
