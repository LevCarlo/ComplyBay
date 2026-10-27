"""Synthetic teleseismic P-wave autocorrelation at the seafloor.

The seafloor transfer function of a plane P wave (:mod:`complybay.forward.response`)
is computed on the FFT grid and passed through the same
:class:`~complybay.acf.ACFProcessing` as the observed traces. Assuming a white
source and noise-free data, the observed ACF estimates this one.

``duration`` sets the frequency spacing. The elastic response rings
(especially in deep water), and anything still ringing after ``duration``
wraps around, so keep it several times longer than the slowest
reverberations you care about.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.fft import next_fast_len, rfftfreq

from complybay.acf.processing import ACFProcessing, acf_from_spectrum
from complybay.forward.response import seafloor_response
from complybay.model import OBSModel

Component = Literal["vertical", "pressure", "radial"]


def synthetic_acf(
    model: OBSModel,
    ray_p: float,
    processing: ACFProcessing,
    *,
    component: Component = "vertical",
    dt: float = 0.05,
    duration: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(lags, ACF) for one component and slowness ``ray_p`` (s/km)."""
    if component == "pressure" and model.water is None:
        raise ValueError("A pressure ACF needs a water layer.")
    if dt <= 0 or duration <= 2 * processing.max_lag:
        raise ValueError("Need dt > 0 and duration longer than twice max_lag.")
    n_fft = next_fast_len(int(round(duration / dt)), real=True)
    freqs = rfftfreq(n_fft, dt)
    spectrum = seafloor_response(model, ray_p, freqs).component(component)
    return acf_from_spectrum(spectrum, dt, n_fft, processing)
