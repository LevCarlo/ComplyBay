"""Assemble an inversion from ``model.yaml`` and (optionally) ``obs.yaml``.

The setup only builds objects; it does not run or write anything. Without
observations it supports prior sampling only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from complybay.inference import (
    JointLikelihood,
    MCMCResult,
    MetropolisConfig,
    PosteriorTarget,
    run_metropolis,
)
from complybay.inference.samplers import ProgressCallback
from complybay.io import load_yaml
from complybay.model import WaterLayer
from complybay.parameterization import (
    DiscretizationConfig,
    GeophysicalPrior,
    ModelParameterization,
    constraints_from_config,
    parameterization_from_config,
    sample_prior,
)
from complybay.workflow.config import (
    MODEL_SECTIONS,
    ObservationSet,
    OutputSettings,
    SamplerSettings,
    check_keys,
    constraint_items,
    discretization_from_config,
    model_parameter_configs,
    observations_from_config,
    output_from_config,
    sampler_from_config,
    water_from_station,
)


@dataclass(frozen=True)
class PriorTarget:
    """Log prior only, as a sampler target (for runs without observations)."""

    prior: GeophysicalPrior

    def log_prior_and_likelihood(self, theta) -> tuple[float, float]:
        value = self.prior.log_prior(theta)
        return (value, 0.0) if np.isfinite(value) else (-np.inf, -np.inf)


@dataclass(frozen=True)
class InversionSetup:
    parameterization: ModelParameterization
    prior: GeophysicalPrior
    discretization: DiscretizationConfig
    sampler: SamplerSettings
    output: OutputSettings
    observations: ObservationSet | None = None
    target: PosteriorTarget | None = None
    station: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return self.parameterization.names

    @property
    def water(self) -> WaterLayer | None:
        return self.parameterization.water

    def metropolis_config(self) -> MetropolisConfig:
        s = self.sampler
        return MetropolisConfig(
            n_steps=s.n_steps,
            proposal_sigma=self.parameterization.proposal_sigmas(),
            bounds=self.parameterization.bounds(),
            burn_in=s.burn_in,
            thin=s.thin,
            update=s.update,
            global_jump_interval=s.global_jump_interval,
            adapt=s.adapt,
            covariance_start=s.covariance_start,
            temperature=s.temperature,
        )

    def initial_thetas(self) -> np.ndarray:
        s = self.sampler
        if s.initial == "theta0":
            return np.tile(self.parameterization.theta0(), (s.n_chains, 1))
        return sample_prior(self.prior, s.n_chains, rng=s.seed).theta

    def run(
        self,
        *,
        prior_only: bool = False,
        progress: ProgressCallback | None = None,
        progress_every: int | None = None,
    ) -> MCMCResult:
        if prior_only:
            target = PriorTarget(self.prior)
        elif self.target is None:
            raise ValueError("A posterior run needs observations (obs.yaml).")
        else:
            target = self.target
        return run_metropolis(
            target,
            self.initial_thetas(),
            self.metropolis_config(),
            seed=self.sampler.seed,
            executor=self.sampler.executor,
            max_workers=self.sampler.max_workers,
            names=self.names,
            progress=progress,
            progress_every=progress_every,
        )


def build_inversion(
    model_cfg: Mapping,
    obs_cfg: Mapping | None = None,
    *,
    obs_dir: str | Path = ".",
) -> InversionSetup:
    """Build an inversion; ``obs_dir`` resolves the data file paths of ``obs_cfg``."""
    check_keys(model_cfg, MODEL_SECTIONS, "model.yaml")
    if "Model" not in model_cfg:
        raise ValueError("model.yaml needs a Model section.")
    if "water" in model_cfg["Model"]:
        raise ValueError(
            "The water column belongs to the station: give it in obs.yaml Station.water."
        )

    parameters = model_parameter_configs(model_cfg)
    observations, water, station = None, None, None
    if obs_cfg is not None:
        observations = observations_from_config(obs_cfg, obs_dir)
        parameters = parameters + list(observations.noise_parameters)
        station_cfg = obs_cfg.get("Station") or {}
        check_keys(station_cfg, {"name", "water"}, "Station")
        water = water_from_station(station_cfg)
        station = station_cfg.get("name")

    model = dict(model_cfg["Model"])
    if water is not None:
        model["water"] = {"depth_km": water.thickness, "vp": water.vp, "rho": water.rho}
    parameterization = parameterization_from_config(
        {"Model": model, "Inversion": {"parameters": parameters}}
    )
    discretization = discretization_from_config(model_cfg.get("Discretization"))
    prior = GeophysicalPrior(
        parameterization,
        discretization,
        constraints_from_config(constraint_items(model_cfg.get("Constraints"))),
    )
    target = None
    if observations is not None:
        target = PosteriorTarget(prior, JointLikelihood(observations.observations))
    return InversionSetup(
        parameterization=parameterization,
        prior=prior,
        discretization=discretization,
        sampler=sampler_from_config(model_cfg.get("Sampler")),
        output=output_from_config(model_cfg.get("Output")),
        observations=observations,
        target=target,
        station=station,
    )


def build_inversion_from_files(model_file: str | Path, obs_file: str | Path | None = None):
    """Read the YAML files and build the inversion (data paths relative to ``obs_file``)."""
    obs_cfg = None if obs_file is None else load_yaml(obs_file)
    obs_dir = "." if obs_file is None else Path(obs_file).parent
    return build_inversion(load_yaml(model_file), obs_cfg, obs_dir=obs_dir)


__all__ = ["InversionSetup", "PriorTarget", "build_inversion", "build_inversion_from_files"]
