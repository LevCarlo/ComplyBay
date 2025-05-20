from numpy.distutils.core import setup, Extension
from setuptools import find_packages

raydep_module = Extension(
    name='ComplyBay.forward.raydep_ft',
    sources=[
        'ComplyBay/forward/raydep_ft.f95',
    ],
)

setup(
    name="ComplyBay",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "bayesbay"
    ],
    ext_modules=[raydep_module],
    author="Mengjie Zheng",
    description="Bayesian MCMC inversion toolkit for seafloor compliance",
    license="MIT",
)