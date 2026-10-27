"""Synthetic teleseismic P-wave autocorrelation at the seafloor.

The seafloor transfer function of a plane P wave (:mod:`complybay.forward.response`)
is computed on the FFT grid and passed through the same
:class:`~complybay.acf.ACFProcessing` as the observed traces. Assuming a white
source and noise-free data, the observed ACF estimates this one.

Only frequencies below :meth:`ACFProcessing.frequency_limit` are computed;
the processing drops the rest for data too.

``duration`` sets the frequency spacing. The elastic response rings
(especially in deep water over a hard seafloor), and whatever is still ringing
after ``duration`` wraps around. :func:`wraparound_fraction` measures how much
of the filtered response is left in the last part of the window; keep it
small (say below 1e-4) for the models and slownesses you invert.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.fft import irfft, next_fast_len, rfftfreq

from complybay.acf.processing import ACFProcessing, acf_from_spectrum
from complybay.forward.response import seafloor_response
from complybay.model import OBSModel

Component = Literal["vertical", "pressure", "radial"]
DEFAULT_DT = 0.05  # s
DEFAULT_DURATION = 200.0  # s; shorter windows wrap in deep water, check first


def _fft_grid(dt: float, duration: float) -> tuple[int, np.ndarray]:
    if dt <= 0 or duration <= 0:
        raise ValueError("dt and duration must be positive.")
    n_fft = next_fast_len(int(round(duration / dt)), real=True)
    return n_fft, rfftfreq(n_fft, dt)


def _band_limited_spectra(model, ray_p, processing, components, freqs):
    """Spectra of ``components`` on ``freqs``, zero above the processing's frequency limit."""
    for component in components:
        if component == "pressure" and model.water is None:
            raise ValueError("A pressure ACF needs a water layer.")
    in_band = freqs <= processing.frequency_limit(freqs)
    response = seafloor_response(model, ray_p, freqs[in_band])
    spectra = {}
    for component in components:
        spectrum = np.zeros(freqs.size, dtype=complex)
        spectrum[in_band] = response.component(component)
        spectra[component] = spectrum
    return spectra


def add_noise_level(spectrum: np.ndarray, n_fft: int, snr: float) -> np.ndarray:
    """Amplitude spectrum with a flat noise level added for signal-to-noise ratio ``snr``.

    Kim et al. (2023, Supporting Text S4): with SNR^2 = sum(s^2 + n^2) / sum(n^2)
    and white noise, Parseval gives the noise amplitude per frequency
    N = sqrt(sum_k |S_k|^2 / (L (SNR^2 - 1))), which is added to |S|. Here the
    sum runs over the two-sided DFT of length L = ``n_fft`` built from the
    one-sided ``spectrum``, restricted to its non-zero (processed) band; the
    synthetic source is a unit impulse. Both are assumptions the paper leaves open.
    """
    if snr <= 1:
        raise ValueError("snr must exceed 1.")
    power = np.abs(spectrum) ** 2
    # Two-sided sum: every one-sided bin twice except zero frequency (and Nyquist).
    two_sided = 2 * power.sum() - power[0] - (power[-1] if n_fft % 2 == 0 else 0.0)
    noise = np.sqrt(two_sided / (n_fft * (snr**2 - 1)))
    return np.where(spectrum != 0, np.abs(spectrum) + noise, 0.0)


def synthetic_acfs(
    model: OBSModel,
    ray_p: float,
    processing: ACFProcessing,
    components: tuple[Component, ...] = ("vertical", "pressure"),
    *,
    dt: float = DEFAULT_DT,
    duration: float = DEFAULT_DURATION,
    snr: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(lags, {component: ACF}) from one seafloor response shared by all components.

    ``snr`` (observed RMS signal-to-noise ratio) adds the flat noise level of
    :func:`add_noise_level` before processing, so the synthetic amplitudes drop
    the way noisy observed ACFs do; ``None`` gives noise-free synthetics.
    """
    if duration <= 2 * processing.max_lag:
        raise ValueError("duration must be longer than twice max_lag.")
    n_fft, freqs = _fft_grid(dt, duration)
    spectra = _band_limited_spectra(model, ray_p, processing, components, freqs)
    acfs = {}
    for component, spectrum in spectra.items():
        if snr is not None:
            spectrum = add_noise_level(spectrum, n_fft, snr)
        lags, acfs[component] = acf_from_spectrum(spectrum, dt, n_fft, processing)
    return lags, acfs


def synthetic_acf(
    model: OBSModel,
    ray_p: float,
    processing: ACFProcessing,
    *,
    component: Component = "vertical",
    dt: float = DEFAULT_DT,
    duration: float = DEFAULT_DURATION,
    snr: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(lags, ACF) for one component and slowness ``ray_p`` (s/km); see :func:`synthetic_acfs`."""
    lags, acfs = synthetic_acfs(
        model, ray_p, processing, (component,), dt=dt, duration=duration, snr=snr
    )
    return lags, acfs[component]


def wraparound_fraction(
    model: OBSModel,
    ray_p: float,
    processing: ACFProcessing,
    *,
    component: Component = "vertical",
    dt: float = DEFAULT_DT,
    duration: float = DEFAULT_DURATION,
    tail: float = 0.1,
) -> float:
    """Energy fraction of the filtered impulse response in the last ``tail`` of ``duration``.

    The synthetic spectrum is periodic in time with period ``duration``; energy
    still arriving near the end of the window is aliased onto early lags. The
    response is filtered with the processing's filters (not whitened) before
    measuring.
    """
    if not 0 < tail < 1:
        raise ValueError("tail must be in (0, 1).")
    n_fft, freqs = _fft_grid(dt, duration)
    spectrum = _band_limited_spectra(model, ray_p, processing, (component,), freqs)[component]
    for f in processing.filters:
        spectrum = spectrum * f.power_response(freqs)
    trace = irfft(np.conj(spectrum), n=n_fft)  # exp(-i omega t) convention -> numpy's
    energy = trace**2
    total = energy.sum()
    if total == 0:
        return 0.0
    return float(energy[int(round((1 - tail) * n_fft)) :].sum() / total)
