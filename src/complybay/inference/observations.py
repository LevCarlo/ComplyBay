"""Observed data groups and how each is predicted from a seafloor model.

Every observation holds data, their standard deviations, and the settings its
forward solver needs, and predicts the same quantity from an
:class:`~complybay.model.OBSModel`. ``noise_parameter`` names the sampled
parameter (log10 of the noise scale h, see :mod:`complybay.inference.likelihood`)
that scales this group's standard deviations; ``None`` keeps them fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from complybay.acf import ACFProcessing
from complybay.forward import compliance, phase_velocity
from complybay.forward.acf import DEFAULT_DT, DEFAULT_DURATION, Component
from complybay.model import OBSModel


def _vector(values, name: str) -> np.ndarray:
    array = np.atleast_1d(np.asarray(values, dtype=float)).copy()
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite 1-D array.")
    array.flags.writeable = False
    return array


def _sigma(sigma, n: int) -> np.ndarray:
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).copy()
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be positive and finite.")
    sigma.flags.writeable = False
    return sigma


@dataclass(frozen=True)
class ComplianceObservation:
    """Normalized compliance (1/Pa) at ``freqs`` (Hz)."""

    freqs: np.ndarray
    values: np.ndarray
    sigma: np.ndarray
    name: str = "compliance"
    noise_parameter: str | None = None
    wavenumber: str = "exact"

    def __post_init__(self):
        freqs = _vector(self.freqs, "freqs")
        values = _vector(self.values, "values")
        if values.size != freqs.size:
            raise ValueError("freqs and values must have the same length.")
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sigma", _sigma(self.sigma, freqs.size))

    def predict(self, model: OBSModel) -> np.ndarray:
        return compliance(model, self.freqs, wavenumber=self.wavenumber)


@dataclass(frozen=True)
class DispersionObservation:
    """Phase velocity (km/s) at ``periods`` (s) of one mode (fundamental Rayleigh by default)."""

    periods: np.ndarray
    values: np.ndarray
    sigma: np.ndarray
    name: str = "dispersion"
    noise_parameter: str | None = None
    wave: str = "rayleigh"
    mode: int = 0

    def __post_init__(self):
        periods = _vector(self.periods, "periods")
        values = _vector(self.values, "values")
        if values.size != periods.size:
            raise ValueError("periods and values must have the same length.")
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sigma", _sigma(self.sigma, periods.size))

    def predict(self, model: OBSModel) -> np.ndarray:
        return phase_velocity(model, self.periods, wave=self.wave, mode=self.mode)


@dataclass(frozen=True)
class ACFObservation:
    """Stacked autocorrelation of one component, on the processing's lag grid.

    ``values`` are on lags 0, dt, ..., max_lag of ``processing`` (as returned
    by :func:`complybay.acf.acf_from_trace` at sampling ``dt``); only lags inside
    ``window`` (s) enter the likelihood. ``ray_p`` (s/km) is the mean ray
    parameter of the stacked events; ``snr`` adds Kim et al. (2023) style noise
    to the synthetics (``None``: noise-free).
    """

    component: Component
    ray_p: float
    processing: ACFProcessing
    values: np.ndarray
    sigma: np.ndarray
    window: tuple[float, float]
    name: str = ""
    noise_parameter: str | None = None
    dt: float = DEFAULT_DT
    duration: float = DEFAULT_DURATION
    snr: float | None = None
    _mask: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.component not in ("vertical", "pressure", "radial"):
            raise ValueError(f"Unknown ACF component {self.component!r}.")
        if not self.name:
            object.__setattr__(self, "name", f"acf_{self.component}")
        values = _vector(self.values, "values")
        n_lags = int(round(self.processing.max_lag / self.dt)) + 1
        if values.size != n_lags:
            raise ValueError(
                f"ACF values need {n_lags} samples (lags 0..max_lag every dt), got {values.size}."
            )
        lo, hi = self.window
        mask = (self.lags >= lo) & (self.lags <= hi)
        if not np.any(mask):
            raise ValueError("The ACF fitting window contains no lags.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sigma", _sigma(self.sigma, values.size))
        object.__setattr__(self, "_mask", mask)

    @property
    def lags(self) -> np.ndarray:
        return np.arange(int(round(self.processing.max_lag / self.dt)) + 1) * self.dt

    @property
    def mask(self) -> np.ndarray:
        """Lags used in the likelihood."""
        return self._mask

    def synthetic_key(self) -> tuple:
        """ACF observations with equal keys share one seafloor response."""
        return (self.ray_p, self.processing, self.dt, self.duration, self.snr)


Observation = ComplianceObservation | DispersionObservation | ACFObservation
