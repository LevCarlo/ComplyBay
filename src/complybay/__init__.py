"""Forward modelling and Bayesian inversion for ocean-bottom seismology."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("complybay")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"
