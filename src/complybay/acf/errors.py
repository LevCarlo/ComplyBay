"""Errors of stacked ACFs: standard deviations and the correlation between lags.

The processing (whitening, filters, normalization) makes the errors of
neighbouring ACF lags strongly correlated: a kim2023 ACF fitted over
0.3-8 s (155 lags) holds about 20 independent pieces of information for
additive noise and 4-11 in simulations where the error comes from
differences between event source wavelets (``docs/inference.md``). Treating
lags as independent overweights the ACF.

Two estimates:

- :func:`stack_errors` (preferred, needs the per-event ACFs): sigma of the
  linear stack, std / sqrt(n_events), and the sample correlation of the event
  ACFs inside a lag window, shrunk towards a target correlation with the
  data-driven weight of Schafer & Strimmer (2005, Stat. Appl. Genet. Mol. Biol.
  4(1):32). With fewer events than lags the sample correlation is singular;
  shrinkage makes it usable. Components of the same events (Z and P) are
  estimated jointly, so their cross-correlation is kept.
- :func:`processing_error_correlation` (fallback, no events needed): the
  correlation implied by the processing for additive noise that is white after
  whitening. To first order the raw sample ACF of such noise has independent
  errors e(k) at lags k > 0 (e(-k) = e(k)); whitening by a running mean over
  W Hz multiplies them by 1 - sinc(W k); the filters convolve with g, the
  inverse transform of their combined power response; the zero-lag taper
  multiplies; and normalization by the ACF at lag k0 adds -m(tau) e(k0) with
  m the expected normalized ACF. Checked against Monte Carlo simulations of
  the processing (``tests/test_acf_errors.py``). Errors caused by the source
  (wavelets that whitening does not flatten) are correlated along the
  structure's own reverberations and are not described; in a simulation with
  random source wavelets the effective number of independent lags was half
  the predicted one or less.

Both return correlations; the likelihood
(:class:`complybay.inference.ErrorCorrelation`) keeps only the leading
eigenvectors, because these matrices are close to singular.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from complybay.acf.processing import ACFProcessing

# Assumed: the filter kernel g is cut where |g| < 1e-6 max |g|.
_KERNEL_CUTOFF = 1e-6


def _filter_kernel(processing: ACFProcessing, dt: float) -> np.ndarray:
    """g(tau) at lags 0, dt, ... until it has decayed (g(-tau) = g(tau))."""
    n_fft = 1 << 16
    freqs = np.fft.rfftfreq(n_fft, dt)
    response = np.ones_like(freqs)
    for f in processing.filters:
        response = response * f.power_response(freqs)
    g = np.fft.irfft(response, n=n_fft)[: n_fft // 2]
    support = np.nonzero(np.abs(g) > _KERNEL_CUTOFF * np.abs(g).max())[0]
    return g[: support[-1] + 1]


def _taper(processing: ACFProcessing, lags: np.ndarray) -> np.ndarray:
    taper = np.ones_like(lags)
    if processing.zero_lag_taper > 0:
        ramp = lags < processing.zero_lag_taper
        taper[ramp] = np.sin(0.5 * np.pi * lags[ramp] / processing.zero_lag_taper) ** 2
    return taper


def processing_error_correlation(
    processing: ACFProcessing, lags, dt: float, *, expected=None
) -> np.ndarray:
    """Correlation of ACF errors at ``lags`` (s, multiples of ``dt``) for additive noise.

    ``expected`` is the expected normalized ACF on lags 0, dt, ...,
    max_lag (the observed stack is the natural estimate); it is required
    unless ``processing.normalize == "none"``. For ``"max"`` normalization the
    peak lag of ``expected`` is assumed not to move between events. Lags where
    the error vanishes (the normalizing lag, a fully tapered zero lag) are not
    allowed.
    """
    lags = np.asarray(lags, dtype=float)
    index = np.rint(lags / dt).astype(int)
    if lags.ndim != 1 or lags.size == 0 or np.any(index < 0):
        raise ValueError("lags must be a non-empty 1-D array of non-negative lags.")
    if not np.allclose(index * dt, lags, atol=1e-6 * dt):
        raise ValueError("lags must be multiples of dt.")
    n_lag = int(round(processing.max_lag / dt)) + 1
    if index.max() >= n_lag:
        raise ValueError("lags beyond the processing's max_lag.")
    normalized = processing.normalize != "none"
    if normalized:
        if expected is None:
            raise ValueError("A normalized ACF needs its expected values (`expected`).")
        expected = np.asarray(expected, dtype=float)
        if expected.shape != (n_lag,):
            raise ValueError(f"expected must have {n_lag} samples (lags 0..max_lag).")
        k0 = 0 if processing.normalize == "zero_lag" else int(np.argmax(np.abs(expected)))
        if expected[k0] == 0:
            raise ValueError("The expected ACF vanishes at its normalizing lag.")
        rows = np.unique(np.append(index, k0))
    else:
        rows = np.unique(index)

    g = _filter_kernel(processing, dt)
    n_k = rows.max() + g.size
    k = np.arange(n_k + 1)
    if processing.whitening is not None:
        weight = (1.0 - np.sinc(processing.whitening.width * k * dt)) ** 2
    else:
        weight = np.ones(k.size)

    def kernel(m):
        m = np.abs(m)
        out = np.zeros(m.shape)
        inside = m < g.size
        out[inside] = g[m[inside]]
        return out

    # error(tau) = sum_k basis[k, tau] e(k), with independent unit e(k) for k >= 0.
    basis = np.sqrt(weight)[:, None] * (
        kernel(rows[None, :] - k[:, None]) + kernel(rows[None, :] + k[:, None])
    )
    basis[0] = np.sqrt(2.0 * weight[0]) * kernel(rows)
    cov = basis.T @ basis
    taper = _taper(processing, rows * dt)
    cov *= np.outer(taper, taper)
    if normalized:
        m = expected[rows] / expected[k0]
        c0 = cov[:, np.searchsorted(rows, k0)]
        cov = (
            cov - np.outer(m, c0) - np.outer(c0, m) + c0[np.searchsorted(rows, k0)] * np.outer(m, m)
        )

    cov = cov[np.ix_(np.searchsorted(rows, index), np.searchsorted(rows, index))]
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if np.any(sd <= 1e-12 * sd.max()):
        raise ValueError("The ACF error vanishes at some lags (normalizing or fully tapered lag).")
    return cov / np.outer(sd, sd)


def shrink_correlation(samples, target) -> tuple[np.ndarray, float]:
    """Sample correlation of ``samples`` (n, p) shrunk towards ``target`` (p, p).

    Returns (correlation, weight lambda) with the estimate of Schafer &
    Strimmer (2005) for lambda: sum Var(r_ij) / sum (r_ij - t_ij)^2 over
    i != j, clipped to [0, 1]. The diagonal stays 1.
    """
    x = np.asarray(samples, dtype=float)
    target = np.asarray(target, dtype=float)
    n, p = x.shape
    if n < 3:
        raise ValueError("Need at least 3 samples to estimate a correlation.")
    if target.shape != (p, p):
        raise ValueError("target must be (p, p) for samples of shape (n, p).")
    std = x.std(axis=0, ddof=1)
    if np.any(std == 0):
        raise ValueError("Some variables do not vary between samples.")
    z = (x - x.mean(axis=0)) / std
    corr = z.T @ z / (n - 1)
    # Var of each r_ij from the products w_kij = z_ki z_kj, without forming them.
    w_mean = corr * (n - 1) / n
    var = n / (n - 1) ** 3 * ((z**2).T @ (z**2) - n * w_mean**2)
    off = ~np.eye(p, dtype=bool)
    denominator = np.sum((corr - target)[off] ** 2)
    lam = 1.0 if denominator == 0 else float(np.clip(var[off].sum() / denominator, 0.0, 1.0))
    shrunk = (1.0 - lam) * corr + lam * target
    np.fill_diagonal(shrunk, 1.0)
    return shrunk, lam


@dataclass(frozen=True)
class StackErrors:
    """Linear stacks of several components with their errors.

    ``stack`` and ``sigma`` are on the full lag grid ``lags`` (0..max_lag);
    ``correlation`` is joint over ``components`` x ``window_lags`` (component
    blocks in the order of ``components``).
    """

    lags: np.ndarray
    components: tuple[str, ...]
    stack: Mapping[str, np.ndarray]
    sigma: Mapping[str, np.ndarray]
    window_lags: np.ndarray
    correlation: np.ndarray
    n_events: int
    shrinkage: float
    target: str
    attrs: Mapping[str, object] = field(default_factory=dict)

    def block(self, components, lags) -> np.ndarray:
        """Correlation restricted to ``components`` (in that order) at ``lags``."""
        index = []
        for component in components:
            c = self.components.index(component)
            positions = np.searchsorted(self.window_lags, lags)
            ok = (positions < self.window_lags.size) & np.isclose(
                self.window_lags[np.minimum(positions, self.window_lags.size - 1)], lags
            )
            if not np.all(ok):
                raise ValueError(f"Lags outside the correlation window of {component!r}.")
            index.extend(c * self.window_lags.size + positions)
        index = np.asarray(index)
        return self.correlation[np.ix_(index, index)]


def stack_errors(
    event_acfs: Mapping[str, np.ndarray],
    dt: float,
    window: tuple[float, float],
    *,
    processing: ACFProcessing | None = None,
) -> StackErrors:
    """Linear stack, sigma and lag correlation from per-event ACFs.

    ``event_acfs`` maps components to arrays (n_events, n_lags) on lags 0, dt,
    ... of the same events in the same order, each processed by the same
    :class:`ACFProcessing` (and normalized per event, as before stacking).
    The correlation is estimated inside ``window`` (s) and shrunk towards
    :func:`processing_error_correlation` of ``processing`` (per component,
    zero between components), or towards the identity without ``processing``.
    Lags whose value does not vary between events (the normalizing zero lag)
    get the median sigma so that the data file stays valid; keep them out of
    the fitting window.
    """
    components = tuple(event_acfs)
    if not components:
        raise ValueError("No components given.")
    arrays = [np.asarray(event_acfs[c], dtype=float) for c in components]
    if any(a.ndim != 2 for a in arrays) or len({a.shape for a in arrays}) != 1:
        raise ValueError("Every component needs an array (n_events, n_lags) of the same shape.")
    n_events, n_lags = arrays[0].shape
    if n_events < 3:
        raise ValueError("Need at least 3 events.")
    lags = np.arange(n_lags) * dt
    inside = (lags >= window[0]) & (lags <= window[1])
    if not np.any(inside):
        raise ValueError("The window contains no lags.")

    stack, sigma = {}, {}
    for component, events in zip(components, arrays, strict=True):
        stack[component] = events.mean(axis=0)
        s = events.std(axis=0, ddof=1) / np.sqrt(n_events)
        zero = s <= 1e-12 * s.max()
        if np.any(zero & inside):
            raise ValueError(f"{component}: ACF does not vary between events inside the window.")
        s[zero] = np.median(s[~zero])
        sigma[component] = s

    p = int(inside.sum())
    target = np.eye(p * len(components))
    if processing is not None:
        if int(round(processing.max_lag / dt)) + 1 != n_lags:
            raise ValueError("event ACFs do not match the processing's max_lag and dt.")
        for c, component in enumerate(components):
            block = slice(c * p, (c + 1) * p)
            target[block, block] = processing_error_correlation(
                processing, lags[inside], dt, expected=stack[component]
            )
    samples = np.concatenate([a[:, inside] for a in arrays], axis=1)
    correlation, lam = shrink_correlation(samples, target)
    return StackErrors(
        lags=lags,
        components=components,
        stack=stack,
        sigma=sigma,
        window_lags=lags[inside],
        correlation=correlation,
        n_events=n_events,
        shrinkage=lam,
        target="processing" if processing is not None else "identity",
    )
