"""Joint Gaussian likelihood with one sampled noise scale per data group.

For data group k with residuals r_i = d_i - g_i(m), standard deviations s_i and
noise scale h_k (a sampled hyperparameter, ``log10 h_k``):

    log L = sum_k [ -1/2 sum_i (r_i / (h_k s_i))^2 - N_k ln h_k ]

Terms that depend on neither the model nor h_k (2 pi, the s_i) are dropped.
The N_k ln h_k term keeps h_k from growing to absorb any misfit, so each group
gets the weight its residuals support instead of a hand-tuned one; at fixed
residuals log L is largest for h_k^2 = chi2_k / N_k. Samples are treated as
independent; correlated ACF lags would need a full covariance.

A forward failure (an exception from a solver, or a non-finite prediction such
as a missing dispersion mode) gives log L = -inf with the reason.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from complybay.forward import synthetic_acfs
from complybay.inference.observations import ACFObservation, Observation
from complybay.model import OBSModel


@dataclass(frozen=True)
class GroupFit:
    name: str
    prediction: np.ndarray  # full prediction (all ACF lags)
    chi2: float  # sum (r / s)^2 over the samples in the likelihood, before scaling
    n: int
    scale: float  # h
    log_likelihood: float


@dataclass(frozen=True)
class LikelihoodEvaluation:
    log_likelihood: float
    groups: tuple[GroupFit, ...] = ()
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def predict(observations: Sequence[Observation], model: OBSModel) -> list[np.ndarray]:
    """Predictions for every observation; ACFs sharing a key share one response."""
    predictions: list[np.ndarray | None] = [None] * len(observations)
    acf_groups: dict[tuple, list[int]] = defaultdict(list)
    for index, obs in enumerate(observations):
        if isinstance(obs, ACFObservation):
            acf_groups[obs.synthetic_key()].append(index)
        else:
            predictions[index] = obs.predict(model)
    for indices in acf_groups.values():
        first = observations[indices[0]]
        components = tuple(dict.fromkeys(observations[i].component for i in indices))
        _, acfs = synthetic_acfs(
            model,
            first.ray_p,
            first.processing,
            components,
            dt=first.dt,
            duration=first.duration,
            snr=first.snr,
        )
        for i in indices:
            predictions[i] = acfs[observations[i].component]
    return predictions


@dataclass(frozen=True)
class JointLikelihood:
    observations: tuple[Observation, ...]

    def __post_init__(self):
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("A likelihood needs at least one observation.")
        names = [o.name for o in observations]
        if len(names) != len(set(names)):
            raise ValueError(f"Observation names must be unique: {names}")
        object.__setattr__(self, "observations", observations)

    @property
    def noise_parameters(self) -> tuple[str, ...]:
        return tuple(o.noise_parameter for o in self.observations if o.noise_parameter)

    def evaluate(
        self, model: OBSModel, parameters: Mapping[str, float] | None = None
    ) -> LikelihoodEvaluation:
        """``parameters`` must hold every ``log10`` noise parameter the observations name."""
        parameters = parameters or {}
        try:
            predictions = predict(self.observations, model)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            return LikelihoodEvaluation(-np.inf, error=f"forward: {exc}")
        groups, total = [], 0.0
        for obs, prediction in zip(self.observations, predictions, strict=True):
            mask = obs.mask if isinstance(obs, ACFObservation) else slice(None)
            residual = (obs.values - prediction)[mask]
            if not np.all(np.isfinite(residual)):
                return LikelihoodEvaluation(-np.inf, error=f"forward: non-finite {obs.name}")
            chi2 = float(np.sum((residual / obs.sigma[mask]) ** 2))
            n = residual.size
            if obs.noise_parameter is None:
                scale = 1.0
            else:
                if obs.noise_parameter not in parameters:
                    raise KeyError(f"Missing noise parameter {obs.noise_parameter!r}.")
                scale = 10.0 ** float(parameters[obs.noise_parameter])
            log_l = -0.5 * chi2 / scale**2 - n * np.log(scale)
            groups.append(GroupFit(obs.name, prediction, chi2, n, scale, log_l))
            total += log_l
        return LikelihoodEvaluation(total, tuple(groups))
