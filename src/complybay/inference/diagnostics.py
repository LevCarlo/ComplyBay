"""Convergence diagnostics for several MCMC chains.

``samples`` are arrays of shape (n_chains, n_draws) for one scalar, or
(n_chains, n_draws, n_parameters).

- Split R-hat (Gelman et al., 2013, Bayesian Data Analysis, 3rd ed., Sec. 11.4):
  each chain is cut in half and the between-half variance is compared with the
  within-half variance. Values near 1 mean all chains sample the same
  distribution; 1.01-1.1 are common thresholds.
- Effective sample size from the multi-chain autocorrelation with Geyer's
  initial positive sequence (Vehtari et al., 2021, Bayesian Analysis 16,
  667-718, without rank normalization): how many independent draws the
  correlated chains are worth.
- Lagging chains: a chain whose log posterior stays below that of the best
  chain (the one with the highest median) without overlap, i.e. its 95th
  percentile below the best chain's 5th percentile (assumed percentiles). In
  a converged run every chain samples the same log-posterior distribution, so
  such a chain is stuck in a lower mode; R-hat flags the run but not which
  chain is at fault. The chain is reported, never dropped: deciding that a
  mode carries no posterior mass needs tempering or many more chains.
"""

from __future__ import annotations

import numpy as np


def _per_parameter(samples, function) -> np.ndarray | float:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 2:
        return function(samples)
    if samples.ndim != 3:
        raise ValueError("samples must have shape (chains, draws) or (chains, draws, parameters).")
    return np.array([function(samples[:, :, i]) for i in range(samples.shape[2])])


def _split_rhat(x: np.ndarray) -> float:
    half = x.shape[1] // 2
    if half < 2:
        raise ValueError("Split R-hat needs at least 4 draws per chain.")
    halves = np.concatenate([x[:, :half], x[:, half : 2 * half]])
    n = halves.shape[1]
    within = halves.var(axis=1, ddof=1).mean()
    between = n * halves.mean(axis=1).var(ddof=1)
    if within == 0:
        return 1.0 if between == 0 else np.inf
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Autocovariance of every chain at lags 0..n-1 (biased estimator, via FFT)."""
    n = x.shape[1]
    centered = x - x.mean(axis=1, keepdims=True)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    return np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :n] / n


def _ess(x: np.ndarray) -> float:
    m, n = x.shape
    if n < 4:
        raise ValueError("ESS needs at least 4 draws per chain.")
    acov = _autocovariance(x)
    within = acov[:, 0].mean() * n / (n - 1)
    var_plus = within * (n - 1) / n + (x.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    if var_plus == 0:
        return float(m * n)
    rho = 1.0 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum pairs rho[2k] + rho[2k+1] while they stay positive (and non-increasing).
    total, previous = 0.0, np.inf
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        pair = min(pair, previous)
        total += pair
        previous = pair
    tau = max(-1.0 + 2.0 * total, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def split_rhat(samples) -> np.ndarray | float:
    """Split R-hat per parameter."""
    return _per_parameter(samples, _split_rhat)


def effective_sample_size(samples) -> np.ndarray | float:
    """Multi-chain effective sample size per parameter."""
    return _per_parameter(samples, _ess)


def lagging_chains(log_posterior) -> tuple[int, ...]:
    """Chains (n_chains, n_draws) whose log posterior lies wholly below the best chain's."""
    lp = np.asarray(log_posterior, dtype=float)
    if lp.ndim != 2 or lp.shape[1] < 2:
        raise ValueError("log_posterior must have shape (chains, draws) with draws >= 2.")
    best = int(np.argmax(np.median(lp, axis=1)))
    floor = np.percentile(lp[best], 5)
    return tuple(int(c) for c in np.nonzero(np.percentile(lp, 95, axis=1) < floor)[0])
