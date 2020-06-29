from setuptools import setup, find_packages
from pathlib import Path

with open(Path(__file__).parent/'README.md', encoding='utf-8') as f:
    long_description = f.read()
setup(
    name="torchvtk",
    description="Efficient data loading and visualization for volumes in PyTorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version="0.2.5",
    packages=find_packages(),
    url='https://github.com/xeTaiz/torchvtk',
    author='Dominik Engel, Marc Mezger',
    install_requires=[
        'torch', 'torchvision', 'numpy', 'matplotlib'
    ],
    dependency_links=[
        'https://github.com/aliutkus/torchsearchsorted.git@master#egg=torchsearchsorted',
        'https://github.com/aliutkus/torchinterp1d.git@master#egg=torchinterp1d'
    ]
)
