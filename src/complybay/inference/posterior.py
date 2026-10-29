"""Posterior target: geophysical prior plus joint likelihood, as a function of theta.

``theta`` holds every sampled parameter: model parameters and the ``log10``
noise scales the observations name. The prior is evaluated first; vectors
outside it get log p = -inf without any forward calculation. With
``prior_only=True`` the target is the prior alone, for sampling the model
space before any data enter (the samplers then explore exactly the prior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from complybay.inference.likelihood import JointLikelihood, LikelihoodEvaluation
from complybay.parameterization import GeophysicalPrior, PriorEvaluation


@dataclass(frozen=True)
class PosteriorEvaluation:
    log_posterior: float
    log_prior: float
    log_likelihood: float
    prior: PriorEvaluation
    likelihood: LikelihoodEvaluation | None = None

    @property
    def error(self) -> str | None:
        if not self.prior.success:
            return self.prior.error
        return None if self.likelihood is None else self.likelihood.error


@dataclass(frozen=True)
class PosteriorTarget:
    prior: GeophysicalPrior
    likelihood: JointLikelihood
    prior_only: bool = False

    def __post_init__(self):
        names = set(self.prior.parameterization.names)
        missing = sorted(set(self.likelihood.noise_parameters) - names)
        if missing:
            raise ValueError(f"Noise parameters are not defined as parameters: {missing}")

    @property
    def names(self) -> tuple[str, ...]:
        return self.prior.parameterization.names

    def evaluate(self, theta) -> PosteriorEvaluation:
        prior = self.prior.evaluate(theta)
        if not prior.success:
            return PosteriorEvaluation(-np.inf, -np.inf, np.nan, prior)
        if self.prior_only:
            return PosteriorEvaluation(prior.log_prior, prior.log_prior, 0.0, prior)
        values = dict(zip(self.names, np.asarray(theta, dtype=float), strict=True))
        likelihood = self.likelihood.evaluate(prior.obs_model, values)
        return PosteriorEvaluation(
            prior.log_prior + likelihood.log_likelihood,
            prior.log_prior,
            likelihood.log_likelihood,
            prior,
            likelihood,
        )

    def log_prior_and_likelihood(self, theta) -> tuple[float, float]:
        """(log prior, log likelihood) for the samplers; (-inf, -inf) outside the prior."""
        result = self.evaluate(theta)
        if not np.isfinite(result.log_prior):
            return -np.inf, -np.inf
        return result.log_prior, result.log_likelihood

    @property
    def detail_names(self) -> tuple[str, ...]:
        """What :meth:`log_prior_likelihood_and_details` returns besides the totals."""
        if self.prior_only:
            return ()
        obs = tuple(f"chi2.{o.name}" for o in self.likelihood.observations)
        return obs + tuple(f"log_likelihood.{t}" for t in self.likelihood.terms)

    def log_prior_likelihood_and_details(self, theta) -> tuple[float, float, np.ndarray]:
        """As :meth:`log_prior_and_likelihood`, plus the per-group values of ``detail_names``."""
        result = self.evaluate(theta)
        if not np.isfinite(result.log_prior):
            return -np.inf, -np.inf, np.full(len(self.detail_names), np.nan)
        details = np.zeros(0) if self.prior_only else self.likelihood.details(result.likelihood)
        return result.log_prior, result.log_likelihood, details

    def __call__(self, theta) -> float:
        """log posterior (log prior with ``prior_only``); what a sampler calls."""
        return self.evaluate(theta).log_posterior
