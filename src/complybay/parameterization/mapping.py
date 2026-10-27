"""Mapping between the sampler's parameter vector and seafloor models.

The sampler only sees ``theta``. :class:`ModelParameterization` owns the
mapping theta -> :class:`ParameterizedModel` -> layers -> :class:`OBSModel`,
independent of any sampler. Adapted from SeisForge
(``seisforge.inv.parameterization``, commit 019d568); the fixed water column
is the ComplyBay addition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from complybay.model import OBSModel, WaterLayer
from complybay.parameterization.config import (
    model_from_config,
    parameter_references,
    validate_model_layout,
    water_from_config,
)
from complybay.parameterization.layering import DiscretizationConfig, discretize
from complybay.parameterization.model import ParameterizedModel
from complybay.parameterization.priors import ParameterSpec, parameter_from_config


@dataclass(frozen=True)
class ModelParameterization:
    """Named scalar parameters plus the ``Model`` mapping that uses them."""

    parameters: tuple[ParameterSpec, ...]
    model_config: Mapping[str, Any]
    water: WaterLayer | None = None

    def __post_init__(self):
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique.")
        validate_model_layout(self.model_config)
        unknown = sorted(parameter_references(self.model_config) - set(names))
        if unknown:
            raise ValueError(f"Model references undefined parameters: {unknown}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def theta0(self) -> np.ndarray:
        return np.array([p.initial for p in self.parameters], dtype=float)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([p.prior.lower for p in self.parameters], dtype=float)
        upper = np.array([p.prior.upper for p in self.parameters], dtype=float)
        return lower, upper

    def proposal_sigmas(self) -> np.ndarray:
        """Per-parameter proposal sigma; 5 % of the prior width when unset."""
        return np.array(
            [
                p.proposal_sigma
                if p.proposal_sigma is not None
                else 0.05 * (p.prior.upper - p.prior.lower)
                for p in self.parameters
            ],
            dtype=float,
        )

    def log_parameter_prior(self, theta) -> float:
        total = 0.0
        for value, p in zip(self._check(theta), self.parameters, strict=True):
            logp = p.prior.logpdf(float(value))
            if not np.isfinite(logp):
                return -np.inf
            total += logp
        return total

    def vector_to_model(self, theta) -> ParameterizedModel:
        return model_from_config(
            self.model_config, dict(zip(self.names, self._check(theta), strict=True))
        )

    def vector_to_obs_model(self, theta, discretization: DiscretizationConfig) -> OBSModel:
        """Layered seafloor model with the fixed water column, ready for every solver."""
        return OBSModel(discretize(self.vector_to_model(theta), discretization), self.water)

    def _check(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.parameters),):
            raise ValueError(f"theta must have shape ({len(self.parameters)},).")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must be finite.")
        return theta


def parameterization_from_config(config: Mapping[str, Any]) -> ModelParameterization:
    """Build from ``{Model: {...}, Inversion: {parameters: [...]}}`` and check theta0."""
    if "Model" not in config:
        raise ValueError("Configuration needs a Model section.")
    model_cfg = config["Model"]
    parameters = tuple(
        parameter_from_config(item) for item in config.get("Inversion", {}).get("parameters", [])
    )
    parameterization = ModelParameterization(
        parameters=parameters,
        model_config={k: v for k, v in model_cfg.items() if k != "water"},
        water=water_from_config(model_cfg),
    )
    try:
        parameterization.vector_to_model(parameterization.theta0())
    except ValueError as exc:
        raise ValueError(f"Initial parameter values give an invalid model: {exc}") from exc
    return parameterization
