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
    ext_modules=[raydep_module],
    author="Mengjie Zheng",
    description="Bayesian MCMC inversion toolkit for seafloor compliance",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "bayesbay"
    ],
    entry_points={
        'console_scripts': [
            'run_comply_inv=ComplyBay.scripts.run_inv:main',
            'run_joint_inv=ComplyBay.scripts.run_joint_inv:main',

        ],
    },
)