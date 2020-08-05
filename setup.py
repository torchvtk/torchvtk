from setuptools import setup, find_packages
from pathlib import Path
from torchvtk import __version__

with open(Path(__file__).parent/'README.md', encoding='utf-8') as f:
    long_description = f.read()
setup(
    name="torchvtk",
    description="Efficient data loading and visualization for volumes in PyTorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=__version__,
    license='MIT',
    packages=find_packages(),
    url='https://github.com/xeTaiz/torchvtk',
    author='Dominik Engel, Marc Mezger',
    install_requires=[
        'torch', 'torchvision',
        'numpy', 'matplotlib', 'psutil',
        'torchsearchsorted @ https://github.com/aliutkus/torchsearchsorted/tarball/master#egg=torchsearchsorted',
        'torchinterp1d @ https://github.com/aliutkus/torchinterp1d/tarball/master#egg=torchinterp1d'
    ],
)
