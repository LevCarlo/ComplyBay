from setuptools import setup, find_packages

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
    author="Mengjie Zheng",
    description="Bayesian MCMC inversion toolkit for seafloor compliance",
    license="MIT",
)