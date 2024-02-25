from setuptools import setup, find_packages

setup(
    name='timanda',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'allantools>=2019.9',
        'astropy>=5.2.1',
        'matplotlib>=3.6.3',
        'numpy>=1.24.1',
        'pandas>=2.2.1',
        'pyqtgraph>=0.13.1',
    ],
)
