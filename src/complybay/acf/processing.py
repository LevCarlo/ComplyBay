"""Autocorrelation (ACF) processing shared by observed and synthetic data.

Every step that changes the shape of an ACF lives in :class:`ACFProcessing`
and is applied identically to a recorded trace (:func:`acf_from_trace`) and to
a synthetic transfer function (:func:`acf_from_spectrum`), so observed and
predicted ACFs are comparable. Steps that only make sense for data (event
selection, stacking) stay outside it.

The flow follows the spectral-whitening autocorrelation of Tauzin, Pham &
Tkalcic (2019, GJI 217, 1-24, doi:10.1093/gji/ggz002):

1. whiten: divide the power spectrum by its running mean over
   ``Whitening.width`` Hz (2N + 1 frequency samples with width = 2N df), or
   divide the amplitude by the running mean of the amplitude
   (``average="amplitude"``); the divisor is floored at ``water_level`` times
   its maximum;
2. autocorrelate: power spectrum, inverse FFT, keep lags 0..``max_lag``;
3. filter: multiply the power spectrum by the zero-phase response of every
   filter in ``filters``, in order (equivalent to filtering the ACF forward
   and backward for the Butterworth filters);
4. taper the zero-lag peak with a sin^2 ramp over ``zero_lag_taper`` s;
5. normalize.

Frequencies the filters suppress below ``band_tolerance`` (relative power) are
dropped before whitening, for data and synthetics alike, so synthetics need
not be computed there (:meth:`ACFProcessing.frequency_limit`).

:data:`PRESETS` holds two flows. ``tauzin2019`` uses unverified starting
values. ``kim2023`` takes the Z-/P-ACF settings of Kim et al. (2023, JGR 128,
e2023JB027616, Fig. 1, Sec. 2.2 and Supporting Text S2); only the order of the
0.5 Hz high-pass is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend, hilbert
from scipy.signal.windows import tukey


@dataclass(frozen=True)
class Whitening:
    """Running-mean spectral whitening.

    ``average="power"`` divides |X|^2 by its running mean (Kim et al., 2023,
    Supporting Eq. S1-S2); ``average="amplitude"`` divides |X| by the running
    mean of |X| and squares the result. ``width`` is the full window width in
    Hz, sampled by 2N + 1 points with N = round(width / (2 df)).
    """

    width: float = 0.5  # Hz
    water_level: float = 0.01  # fraction of the maximum running mean
    average: Literal["power", "amplitude"] = "amplitude"

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("Whitening width must be positive.")
        if not 0 <= self.water_level < 1:
            raise ValueError("water_level must be in [0, 1).")
        if self.average not in ("power", "amplitude"):
            raise ValueError(f"Unknown whitening average {self.average!r}.")

    def whitened_power(self, amplitude: np.ndarray, df: float, in_band: np.ndarray) -> np.ndarray:
        """Whitened power spectrum from the amplitude spectrum on a grid of spacing ``df``.

        The water level refers to the maximum running mean inside ``in_band``.
        """
        half = int(round(self.width / (2 * df)))
        values = amplitude**2 if self.average == "power" else amplitude
        smooth = uniform_filter1d(values, size=2 * half + 1, mode="nearest")
        floor = max(self.water_level * smooth[in_band].max(), np.finfo(float).tiny)
        whitened = values / np.maximum(smooth, floor)
        return whitened if self.average == "power" else whitened**2


class Filter(Protocol):
    """Anything with a zero-phase response to multiply into the power spectrum."""

    def power_response(self, freqs: np.ndarray) -> np.ndarray: ...


def _butter_high(freqs, corner, order):
    # |H|^2 of a Butterworth high-pass of the given order.
    return 1.0 / (1.0 + (corner / np.maximum(freqs, 1e-12)) ** (2 * order))


def _butter_low(freqs, corner, order):
    return 1.0 / (1.0 + (freqs / corner) ** (2 * order))


def _check_corners(corners):
    if corners < 1:
        raise ValueError("corners must be at least 1.")


@dataclass(frozen=True)
class Bandpass:
    """Butterworth band-pass applied with zero phase (|B(f)|^2 on the power spectrum)."""

    freqmin: float
    freqmax: float
    corners: int = 2

    def __post_init__(self):
        if not 0 < self.freqmin < self.freqmax:
            raise ValueError("Need 0 < freqmin < freqmax.")
        _check_corners(self.corners)

    def power_response(self, freqs: np.ndarray) -> np.ndarray:
        high = _butter_high(freqs, self.freqmin, self.corners)
        low = _butter_low(freqs, self.freqmax, self.corners)
        return (high * low) ** 2


@dataclass(frozen=True)
class Highpass:
    """Butterworth high-pass applied with zero phase."""

    freq: float
    corners: int = 2

    def __post_init__(self):
        if self.freq <= 0:
            raise ValueError("High-pass corner must be positive.")
        _check_corners(self.corners)

    def power_response(self, freqs: np.ndarray) -> np.ndarray:
        return _butter_high(freqs, self.freq, self.corners) ** 2


@dataclass(frozen=True)
class Lowpass:
    """Butterworth low-pass applied with zero phase."""

    freq: float
    corners: int = 2

    def __post_init__(self):
        if self.freq <= 0:
            raise ValueError("Low-pass corner must be positive.")
        _check_corners(self.corners)

    def power_response(self, freqs: np.ndarray) -> np.ndarray:
        return _butter_low(freqs, self.freq, self.corners) ** 2


@dataclass(frozen=True)
class GaussianFilter:
    """Gaussian low-pass exp(-(omega / 2a)^2), as used for receiver functions."""

    a: float

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("Gaussian a must be positive.")

    def power_response(self, freqs: np.ndarray) -> np.ndarray:
        return np.exp(-((2 * np.pi * freqs / (2 * self.a)) ** 2))


Normalization = Literal["max", "zero_lag", "none"]


@dataclass(frozen=True)
class ACFProcessing:
    """Waveform-shaping steps of the ACF, shared by data and synthetics."""

    max_lag: float = 30.0  # s
    whitening: Whitening | None = field(default_factory=Whitening)
    filters: tuple[Filter, ...] = ()  # applied in order; a single filter is accepted
    zero_lag_taper: float = 0.5  # s; 0 disables
    normalize: Normalization = "max"
    band_tolerance: float = 1e-6  # drop frequencies where the filters pass less power

    def __post_init__(self):
        filters = self.filters
        if filters is None:
            filters = ()
        elif hasattr(filters, "power_response"):
            filters = (filters,)
        filters = tuple(filters)
        for f in filters:
            if not callable(getattr(f, "power_response", None)):
                raise TypeError(f"{f!r} has no power_response(freqs) method.")
        object.__setattr__(self, "filters", filters)
        if self.max_lag <= 0:
            raise ValueError("max_lag must be positive.")
        if self.zero_lag_taper < 0:
            raise ValueError("zero_lag_taper must be non-negative.")
        if self.normalize not in ("max", "zero_lag", "none"):
            raise ValueError(f"Unknown normalization {self.normalize!r}.")
        if not 0 <= self.band_tolerance < 1:
            raise ValueError("band_tolerance must be in [0, 1).")

    def frequency_limit(self, freqs: np.ndarray) -> float:
        """Highest frequency (Hz) that can affect the ACF on the grid ``freqs``.

        Above the last frequency where the combined filter power is at least
        ``band_tolerance`` times its peak, plus half a whitening window (so the
        running mean inside the band is unchanged), the spectrum is dropped.
        Without filters, or with ``band_tolerance = 0``, nothing is dropped.
        """
        if not self.filters or self.band_tolerance == 0:
            return np.inf
        response = np.ones_like(freqs)
        for f in self.filters:
            response = response * f.power_response(freqs)
        passing = np.nonzero(response >= self.band_tolerance * response.max())[0]
        margin = 0.5 * self.whitening.width if self.whitening is not None else 0.0
        return float(freqs[passing[-1]] + margin)


PRESETS: dict[str, ACFProcessing] = {
    # Unverified starting values for a Tauzin et al. (2019)-style flow.
    "tauzin2019": ACFProcessing(
        max_lag=30.0,
        whitening=Whitening(width=0.5, water_level=0.01),
        filters=(Bandpass(freqmin=0.5, freqmax=4.0, corners=2),),
        zero_lag_taper=0.5,
        normalize="max",
    ),
    # Kim et al. (2023) Z-/P-ACF (Supporting Text S2): power spectrum divided by its
    # running mean over W = 0.8 Hz, no water level, Gaussian a = 5, then a 0.5 Hz
    # high-pass after autocorrelation, normalized at t = 0; zero lag left in place
    # (they exclude it from the fitting window). The high-pass order is assumed.
    # Their data window: 70 s from 10 s before P, Tukey cosine fraction 0.2
    # (acf_from_trace(..., taper_fraction=0.2)).
    "kim2023": ACFProcessing(
        max_lag=30.0,
        whitening=Whitening(width=0.8, water_level=0.0, average="power"),
        filters=(GaussianFilter(a=5.0), Highpass(freq=0.5, corners=2)),
        zero_lag_taper=0.0,
        normalize="zero_lag",
    ),
}


def preset(name: str) -> ACFProcessing:
    """A named processing preset, see :data:`PRESETS`."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}.") from None


def acf_from_spectrum(
    spectrum, dt: float, n_fft: int, processing: ACFProcessing
) -> tuple[np.ndarray, np.ndarray]:
    """ACF (lags, values) from a one-sided spectrum on the ``rfft`` grid of ``n_fft``, ``dt``."""
    spectrum = np.asarray(spectrum, dtype=complex)
    if spectrum.size != n_fft // 2 + 1:
        raise ValueError(f"Spectrum needs {n_fft // 2 + 1} samples for n_fft={n_fft}.")
    df = 1.0 / (n_fft * dt)
    freqs = np.arange(spectrum.size) * df
    in_band = freqs <= processing.frequency_limit(freqs)

    amplitude = np.where(in_band, np.abs(spectrum), 0.0)
    if processing.whitening is not None:
        power = processing.whitening.whitened_power(amplitude, df, in_band)
    else:
        power = amplitude**2
    for f in processing.filters:
        power = power * f.power_response(freqs)

    acf = irfft(power, n=n_fft)
    n_lag = int(round(processing.max_lag / dt))
    if n_lag >= n_fft // 2:
        raise ValueError("max_lag must be shorter than half the FFT length.")
    lags = np.arange(n_lag + 1) * dt
    acf = acf[: n_lag + 1].copy()
    zero_lag = acf[0]

    if processing.zero_lag_taper > 0:
        ramp = lags < processing.zero_lag_taper
        acf[ramp] *= np.sin(0.5 * np.pi * lags[ramp] / processing.zero_lag_taper) ** 2

    if processing.normalize == "max":
        peak = np.max(np.abs(acf))
        if peak > 0:
            acf /= peak
    elif processing.normalize == "zero_lag" and zero_lag != 0:
        acf /= zero_lag
    return lags, acf


def acf_from_trace(
    trace, dt: float, processing: ACFProcessing, *, taper_fraction: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """ACF of one recorded trace (already cut to the P-coda window).

    The trace is demeaned, detrended and Tukey-tapered, then zero-padded to at
    least twice its length so the autocorrelation does not wrap around.
    """
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1 or trace.size < 2:
        raise ValueError("trace must be a 1-D array with at least two samples.")
    x = detrend(trace - trace.mean(), type="linear") * tukey(trace.size, taper_fraction)
    n_fft = next_fast_len(2 * trace.size)
    return acf_from_spectrum(rfft(x, n=n_fft), dt, n_fft, processing)


def stack_acfs(acfs, *, method: Literal["linear", "pws"] = "linear", power: float = 2.0):
    """Stack ACFs of shape (n_events, n_lags); ``pws`` is phase-weighted stacking."""
    acfs = np.asarray(acfs, dtype=float)
    if acfs.ndim != 2:
        raise ValueError("acfs must have shape (n_events, n_lags).")
    linear = acfs.mean(axis=0)
    if method == "linear":
        return linear
    if method != "pws":
        raise ValueError(f"Unknown stacking method {method!r}.")
    analytic = hilbert(acfs, axis=1)
    coherence = np.abs(np.mean(analytic / np.maximum(np.abs(analytic), 1e-300), axis=0))
    return linear * coherence**power
