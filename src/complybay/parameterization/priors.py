"""Scalar parameter priors and parameter specifications.

Adapted from SeisForge (``seisforge.inv.parameterization``, commit 019d568).
Every prior has hard bounds: they are the support the sampler explores and
draws starting points from; the prior type sets the density inside them. Log
densities are unnormalized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import truncnorm


def _check_bounds(lower: float, upper: float) -> None:
    if not (np.isfinite(lower) and np.isfinite(upper)) or upper <= lower:
        raise ValueError("Prior bounds must be finite with lower < upper.")


@dataclass(frozen=True)
class UniformPrior:
    lower: float
    upper: float

    def __post_init__(self):
        _check_bounds(self.lower, self.upper)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def logpdf(self, value: float) -> float:
        return 0.0 if self.contains(value) else -np.inf

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lower, self.upper))


@dataclass(frozen=True)
class NormalPrior:
    """Gaussian truncated to [lower, upper]; e.g. an interface depth from another method."""

    mean: float
    sigma: float
    lower: float
    upper: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Normal prior needs a finite mean and positive sigma.")
        _check_bounds(self.lower, self.upper)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def logpdf(self, value: float) -> float:
        if not self.contains(value):
            return -np.inf
        return float(-0.5 * ((value - self.mean) / self.sigma) ** 2)

    def draw(self, rng: np.random.Generator) -> float:
        a, b = (self.lower - self.mean) / self.sigma, (self.upper - self.mean) / self.sigma
        return float(truncnorm.rvs(a, b, loc=self.mean, scale=self.sigma, random_state=rng))


@dataclass(frozen=True)
class HalfNormalPrior:
    """Half-normal on [lower, upper] peaking at ``lower``.

    Used for transition widths: a hard step is the default and a gradational
    interface has to improve the fit enough to pay for its prior cost.
    """

    scale: float
    upper: float
    lower: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Half-normal prior scale must be finite and positive.")
        _check_bounds(self.lower, self.upper)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def logpdf(self, value: float) -> float:
        if not self.contains(value):
            return -np.inf
        return float(-0.5 * ((value - self.lower) / self.scale) ** 2)

    def draw(self, rng: np.random.Generator) -> float:
        b = (self.upper - self.lower) / self.scale
        return float(truncnorm.rvs(0.0, b, loc=self.lower, scale=self.scale, random_state=rng))


ParameterPrior = UniformPrior | NormalPrior | HalfNormalPrior


@dataclass(frozen=True)
class ParameterSpec:
    """One sampled scalar. ``proposal_sigma`` tunes the sampler only, not the prior."""

    name: str
    initial: float
    prior: ParameterPrior
    proposal_sigma: float | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Parameter name cannot be empty.")
        if not np.isfinite(self.initial) or not self.prior.contains(self.initial):
            raise ValueError(f"Initial value of {self.name} must lie inside its prior bounds.")
        if self.proposal_sigma is not None and self.proposal_sigma <= 0:
            raise ValueError(f"Proposal sigma of {self.name} must be positive.")


def prior_from_config(config: Mapping[str, Any]) -> ParameterPrior:
    """``{type: uniform|normal|half_normal, bounds: [lower, upper], ...}``."""
    prior_type = str(config.get("type", "uniform")).lower()
    bounds = config.get("bounds")
    if bounds is None:
        raise ValueError("Every parameter prior needs bounds: [lower, upper].")
    lower, upper = (float(v) for v in bounds)
    if prior_type == "uniform":
        return UniformPrior(lower=lower, upper=upper)
    if prior_type == "normal":
        return NormalPrior(
            mean=float(config["mean"]), sigma=float(config["sigma"]), lower=lower, upper=upper
        )
    if prior_type == "half_normal":
        return HalfNormalPrior(scale=float(config["scale"]), lower=lower, upper=upper)
    raise ValueError(f"Unsupported prior type: {config.get('type')!r}")


def parameter_from_config(config: Mapping[str, Any]) -> ParameterSpec:
    """``{name, initial, prior: {...}, proposal: {sigma}}``."""
    sigma = config.get("proposal", {}).get("sigma")
    return ParameterSpec(
        name=str(config["name"]),
        initial=float(config["initial"]),
        prior=prior_from_config(config.get("prior", {})),
        proposal_sigma=None if sigma is None else float(sigma),
    )
