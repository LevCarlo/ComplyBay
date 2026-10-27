"""The geophysical prior: parameter priors plus the support of valid models.

Adapted from SeisForge (``seisforge.inv.prior``, commit 019d568). A parameter
vector is inside the prior only if

1. every parameter lies inside its prior bounds,
2. it maps to a valid model: interfaces in order, transition supports not
   overlapping, every relation inside its range of validity, and positive
   elastic moduli in every layer (``model_mapping``),
3. the model passes every hard constraint (``physical_constraints``).

Failures return log prior = -inf with the reason; they never raise, so a
sampler can propose anywhere. A valid evaluation keeps the built
:class:`~complybay.model.OBSModel`, so the likelihood does not rebuild it.
Soft priors (SeisForge's curvature prior) are not ported yet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from complybay.model import OBSModel
from complybay.parameterization.constraints import ConstraintSuite
from complybay.parameterization.layering import DiscretizationConfig, discretize
from complybay.parameterization.mapping import ModelParameterization
from complybay.parameterization.model import ParameterizedModel


@dataclass(frozen=True)
class PriorEvaluation:
    log_prior: float
    error: str | None = None  # None when the model is inside the prior
    model: ParameterizedModel | None = None
    obs_model: OBSModel | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class GeophysicalPrior:
    parameterization: ModelParameterization
    discretization: DiscretizationConfig
    constraints: ConstraintSuite = field(default_factory=ConstraintSuite)

    def __post_init__(self):
        names = {item.get("name") for item in self.parameterization.model_config["segments"]}
        for constraint in self.constraints.constraints:
            unknown = sorted(set(getattr(constraint, "segments", None) or ()) - names)
            if unknown:
                raise ValueError(f"Constraint {constraint.name} names unknown segments: {unknown}")

    def evaluate(self, theta) -> PriorEvaluation:
        log_prior = self.parameterization.log_parameter_prior(theta)
        if not np.isfinite(log_prior):
            return PriorEvaluation(-np.inf, "parameter_prior")
        try:
            model = self.parameterization.vector_to_model(theta)
            obs = OBSModel(discretize(model, self.discretization), self.parameterization.water)
        except ValueError as exc:
            return PriorEvaluation(-np.inf, f"model_mapping: {exc}")
        result = self.constraints.evaluate(model, obs)
        if not result.valid:
            return PriorEvaluation(
                -np.inf, "physical_constraints: " + ", ".join(result.failed), model, obs
            )
        return PriorEvaluation(log_prior, None, model, obs)

    def log_prior(self, theta) -> float:
        return self.evaluate(theta).log_prior


@dataclass(frozen=True)
class PriorSamples:
    """Independent draws from the prior, by rejection."""

    theta: np.ndarray  # (n_samples, n_parameters)
    n_attempts: int
    failures: dict[str, int]  # rejection reason (category) -> count

    @property
    def acceptance(self) -> float:
        return self.theta.shape[0] / self.n_attempts


def sample_prior(
    prior: GeophysicalPrior,
    n_samples: int,
    *,
    rng: np.random.Generator | int | None = None,
    max_attempts: int | None = None,
) -> PriorSamples:
    """Draw parameters from their priors and keep the vectors inside the full prior.

    The kept vectors are exact independent samples of the prior (parameter
    priors restricted to the valid-model support). The acceptance rate and the
    rejection reasons show how much of the parameter box the constraints
    remove; a very low acceptance means bounds and constraints contradict each
    other. Use the samples as diverse chain starting points, or plot their
    models to see the model space before any data enters.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(rng)
    max_attempts = 1000 * n_samples if max_attempts is None else max_attempts
    parameters = prior.parameterization.parameters
    kept, failures, attempts = [], Counter(), 0
    while len(kept) < n_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Only {len(kept)} of {n_samples} prior samples after {attempts} attempts; "
                f"rejections: {dict(failures)}"
            )
        attempts += 1
        theta = np.array([p.prior.draw(rng) for p in parameters])
        result = prior.evaluate(theta)
        if result.success:
            kept.append(theta)
        else:
            failures[result.error.split(":")[0]] += 1
    return PriorSamples(np.array(kept), attempts, dict(failures))
