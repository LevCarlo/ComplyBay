"""Parse ``model.yaml`` and ``obs.yaml`` into the scientific objects.

``model.yaml`` describes the model space and how it is sampled::

    Model: {...}              # segments, interfaces, relations (no water)
    Inversion: {parameters: [...]}
    Constraints: [...]        # or {constraints: [...]}
    Discretization: {...}
    Sampler: {...}
    Output: {...}

``obs.yaml`` describes one station, its data and the forward settings::

    Station: {name: OBS01, water: {depth_km: 2.0}}
    Observations:
      compliance: {file: compliance.dat, noise: {bounds: [-0.5, 0.5]}}
      dispersion: {file: vph.dat, wave: rayleigh, mode: 0, noise: {...}}
      acf:
        processing: kim2023          # preset name, {preset: ..., <overrides>}, or full settings
        ray_p: 0.06                  # s/km, mean of the stacked events
        window: [0.3, 8.0]           # s, lags that enter the likelihood
        snr: null                    # Kim et al. (2023) noise level in the synthetics
        error_correlation: none      # or processing, or {file: acf_errors.nc}; see below
        components:
          vertical: {file: acf_z.dat, noise: {...}}
          pressure: {file: acf_p.dat, noise: {...}}
    Forward:
      compliance: {wavenumber: exact}
      dispersion: {algorithm: dunkin, dc: 0.005}
      acf: {duration: 200.0}

The water column belongs to the station, so it is given in ``obs.yaml`` and
the same ``model.yaml`` serves stations at different depths. Each data group
with ``noise`` gets a sampled ``noise.<group>`` parameter (log10 of its noise
scale; defaults: bounds [-0.5, 0.5], initial 0, proposal sigma 0.05); without
``noise`` its sigma is used as given. Relative file paths are relative to the
YAML file. Unknown keys are errors, so typos do not pass silently.

``Observations.acf.error_correlation`` sets how ACF errors are correlated
between lags (:mod:`complybay.acf.errors`, :class:`ErrorCorrelation`):
``none`` (default, independent lags), ``processing`` (theory for additive
noise, per component, with the observed ACF as the expected one), or
``{file: acf_errors.nc}`` (written by :func:`complybay.io.write_acf_errors`
from the event ACFs; one joint block for all components, so Z-P correlation
is kept). Either mapping form takes ``variance_fraction`` (default 0.99).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from complybay.acf import (
    ACFProcessing,
    Bandpass,
    GaussianFilter,
    Highpass,
    Lowpass,
    Whitening,
    preset,
    processing_error_correlation,
)
from complybay.inference import (
    ACFObservation,
    ComplianceObservation,
    DispersionObservation,
    ErrorCorrelation,
    Observation,
    geometric_temperatures,
)
from complybay.inference.likelihood import DEFAULT_VARIANCE_FRACTION
from complybay.io import read_acf_errors, read_three_columns
from complybay.model import WaterLayer
from complybay.parameterization import DiscretizationConfig

MODEL_SECTIONS = {"Model", "Inversion", "Constraints", "Discretization", "Sampler", "Output"}
OBS_SECTIONS = {"Station", "Observations", "Forward"}
NOISE_DEFAULTS = {"bounds": [-0.5, 0.5], "initial": 0.0, "proposal_sigma": 0.05}


def check_keys(mapping: Mapping, allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; allowed: {sorted(allowed)}.")


# --- model.yaml -------------------------------------------------------------------


def discretization_from_config(config: Mapping | None) -> DiscretizationConfig:
    config = dict(config or {})
    check_keys(
        config,
        {"dz", "layers_per_segment", "transition_layers", "layer_value", "zmax"},
        "Discretization",
    )
    return DiscretizationConfig(**config)


def constraint_items(config) -> list:
    if config is None:
        return []
    if isinstance(config, Mapping):
        check_keys(config, {"constraints"}, "Constraints")
        return list(config.get("constraints") or [])
    return list(config)


@dataclass(frozen=True)
class SamplerSettings:
    n_steps: int = 20000
    burn_in: int = 5000
    thin: int = 10
    n_chains: int = 4
    update: str = "one"
    adapt: bool = True
    covariance_start: int | None = None  # update: covariance; default burn_in // 2
    global_jump_interval: int | None = 20
    temperature: float = 1.0
    tempering: Mapping | None = None  # parallel tempering, see tempering_ladder
    executor: str = "process"
    max_workers: int | None = None
    seed: int | None = None
    initial: str = (
        "prior"  # "prior": exact prior samples; "theta0": all chains at the initial values
    )

    def __post_init__(self):
        if self.n_chains < 1:
            raise ValueError("Sampler.n_chains must be positive.")
        if self.initial not in ("prior", "theta0"):
            raise ValueError("Sampler.initial must be 'prior' or 'theta0'.")
        if self.executor not in ("serial", "thread", "process"):
            raise ValueError("Sampler.executor must be serial, thread or process.")
        _ = self.tempering_ladder  # validates Sampler.tempering

    @property
    def tempering_ladder(self) -> tuple[tuple[float, ...] | None, int]:
        """(temperatures, swap_interval) from ``Sampler.tempering``.

        ``tempering: {n_temperatures: 5, max_temperature: 30, swap_interval: 1}``
        (a geometric ladder; the values shown are the defaults, assumed) or
        ``{temperatures: [1, 2, 5, ...]}``.
        """
        if self.tempering is None:
            return None, 1
        t = dict(self.tempering)
        check_keys(
            t, {"n_temperatures", "max_temperature", "temperatures", "swap_interval"}, "tempering"
        )
        if "temperatures" in t:
            if {"n_temperatures", "max_temperature"} & set(t):
                raise ValueError("Sampler.tempering: give temperatures or a geometric ladder.")
            ladder = tuple(float(v) for v in t["temperatures"])
        else:
            ladder = geometric_temperatures(
                int(t.get("n_temperatures", 5)), float(t.get("max_temperature", 30.0))
            )
        return ladder, int(t.get("swap_interval", 1))


def sampler_from_config(config: Mapping | None) -> SamplerSettings:
    config = dict(config or {})
    check_keys(config, {f.name for f in dataclasses.fields(SamplerSettings)}, "Sampler")
    return SamplerSettings(**config)


@dataclass(frozen=True)
class OutputSettings:
    max_depth_km: float | None = None  # depth range of the model ensemble; None: model zmax
    n_depths: int = 401
    n_model_samples: int = 2000  # posterior samples kept in the model ensemble
    n_predictive_samples: int = 200  # posterior samples run through the forward solvers

    def __post_init__(self):
        if self.n_depths < 2 or self.n_model_samples < 1 or self.n_predictive_samples < 1:
            raise ValueError("Output sample counts must be positive (n_depths >= 2).")


def output_from_config(config: Mapping | None) -> OutputSettings:
    config = dict(config or {})
    check_keys(config, {f.name for f in dataclasses.fields(OutputSettings)}, "Output")
    return OutputSettings(**config)


# --- obs.yaml ---------------------------------------------------------------------


def water_from_station(station: Mapping | None) -> WaterLayer | None:
    if not station or station.get("water") is None:
        return None
    water = dict(station["water"])
    check_keys(water, {"depth_km", "vp", "rho"}, "Station.water")
    return WaterLayer(
        float(water["depth_km"]), float(water.get("vp", 1.5)), float(water.get("rho", 1.03))
    )


def _filter_from_config(config: Mapping):
    config = dict(config)
    kind = str(config.pop("type")).lower()
    classes = {
        "bandpass": Bandpass,
        "highpass": Highpass,
        "lowpass": Lowpass,
        "gaussian": GaussianFilter,
    }
    if kind not in classes:
        raise ValueError(f"Unknown ACF filter type {kind!r}; choose from {sorted(classes)}.")
    return classes[kind](**config)


def processing_from_config(config, *, max_lag: float) -> ACFProcessing:
    """A preset name, ``{preset: name, <overrides>}``, or a full mapping; max_lag from the data."""
    if isinstance(config, str):
        base, overrides = preset(config), {}
    else:
        overrides = dict(config)
        name = overrides.pop("preset", None)
        base = preset(name) if name else ACFProcessing()
    check_keys(
        overrides,
        {"whitening", "filters", "zero_lag_taper", "normalize", "band_tolerance"},
        "Observations.acf.processing",
    )
    if "whitening" in overrides:
        w = overrides["whitening"]
        overrides["whitening"] = None if w is None else Whitening(**w)
    if "filters" in overrides:
        overrides["filters"] = tuple(_filter_from_config(f) for f in overrides["filters"] or ())
    return dataclasses.replace(base, max_lag=max_lag, **overrides)


def _noise(group: str, config, parameters: list[dict]) -> str | None:
    """Register ``noise.<group>`` if ``config`` asks for it; return its name."""
    if config is None or config is False:
        return None
    settings = dict(NOISE_DEFAULTS)
    if isinstance(config, Mapping):
        check_keys(config, set(NOISE_DEFAULTS), f"noise of {group}")
        settings.update(config)
    elif config is not True:
        raise ValueError(f"noise of {group} must be true, a mapping, or absent.")
    name = f"noise.{group}"
    parameters.append(
        {
            "name": name,
            "initial": float(settings["initial"]),
            "prior": {"bounds": [float(b) for b in settings["bounds"]]},
            "proposal": {"sigma": float(settings["proposal_sigma"])},
        }
    )
    return name


@dataclass(frozen=True)
class ObservationSet:
    observations: tuple[Observation, ...]
    noise_parameters: tuple[dict, ...]  # parameter configs appended to the model parameters
    files: dict[str, str] = field(default_factory=dict)  # group -> data file
    correlations: tuple[ErrorCorrelation, ...] = ()


def observations_from_config(obs_cfg: Mapping, base_dir: str | Path) -> ObservationSet:
    base_dir = Path(base_dir)
    check_keys(obs_cfg, OBS_SECTIONS, "obs.yaml")
    groups = dict(obs_cfg.get("Observations") or {})
    check_keys(groups, {"compliance", "dispersion", "acf"}, "Observations")
    forward = dict(obs_cfg.get("Forward") or {})
    check_keys(forward, {"compliance", "dispersion", "acf"}, "Forward")
    observations, parameters, files = [], [], {}

    def path(item):
        return (base_dir / item["file"]).resolve()

    if "compliance" in groups:
        item = dict(groups["compliance"])
        check_keys(item, {"file", "noise"}, "Observations.compliance")
        settings = dict(forward.get("compliance") or {})
        check_keys(settings, {"wavenumber"}, "Forward.compliance")
        data = read_three_columns(path(item))
        noise = _noise("compliance", item.get("noise"), parameters)
        observations.append(
            ComplianceObservation(data.x, data.value, data.sigma, noise_parameter=noise, **settings)
        )
        files["compliance"] = str(data.path)

    if "dispersion" in groups:
        item = dict(groups["dispersion"])
        check_keys(item, {"file", "noise", "wave", "mode"}, "Observations.dispersion")
        settings = dict(forward.get("dispersion") or {})
        check_keys(settings, {"algorithm", "dc"}, "Forward.dispersion")
        data = read_three_columns(path(item))
        noise = _noise("dispersion", item.get("noise"), parameters)
        observations.append(
            DispersionObservation(
                data.x,
                data.value,
                data.sigma,
                noise_parameter=noise,
                wave=item.get("wave", "rayleigh"),
                mode=int(item.get("mode", 0)),
                **settings,
            )
        )
        files["dispersion"] = str(data.path)

    if "acf" in groups:
        item = dict(groups["acf"])
        check_keys(
            item,
            {"processing", "ray_p", "window", "snr", "components", "error_correlation"},
            "Observations.acf",
        )
        settings = dict(forward.get("acf") or {})
        check_keys(settings, {"duration"}, "Forward.acf")
        components = dict(item.get("components") or {})
        if not components:
            raise ValueError("Observations.acf needs at least one component.")
        for component, comp_cfg in components.items():
            comp_cfg = dict(comp_cfg)
            check_keys(comp_cfg, {"file", "noise"}, f"Observations.acf.components.{component}")
            data = read_three_columns(path(comp_cfg))
            dt = float(np.mean(np.diff(data.x)))
            if data.x[0] != 0 or not np.allclose(np.diff(data.x), dt, rtol=1e-6, atol=1e-9):
                raise ValueError(f"{data.path}: ACF lags must start at 0 with a constant step.")
            processing = processing_from_config(
                item.get("processing", "kim2023"), max_lag=data.x[-1]
            )
            group = f"acf_{component}"
            observations.append(
                ACFObservation(
                    component=component,
                    ray_p=float(item["ray_p"]),
                    processing=processing,
                    values=data.value,
                    sigma=data.sigma,
                    window=tuple(float(v) for v in item["window"]),
                    name=group,
                    noise_parameter=_noise(group, comp_cfg.get("noise"), parameters),
                    dt=dt,
                    snr=None if item.get("snr") is None else float(item["snr"]),
                    **({"duration": float(settings["duration"])} if "duration" in settings else {}),
                )
            )
            files[group] = str(data.path)
        acfs = [o for o in observations if isinstance(o, ACFObservation)]
        correlations, source = _acf_correlations(item.get("error_correlation"), acfs, path)
        if source is not None:
            files["acf_errors"] = source
    else:
        correlations = []

    if not observations:
        raise ValueError("obs.yaml defines no observations.")
    return ObservationSet(tuple(observations), tuple(parameters), files, tuple(correlations))


def _acf_correlations(config, acfs: list[ACFObservation], path) -> tuple[list, str | None]:
    """ErrorCorrelation blocks for ``Observations.acf.error_correlation``."""
    if config is None or config == "none":
        return [], None
    if config == "processing":
        config = {"model": "processing"}
    if not isinstance(config, Mapping):
        raise ValueError("Observations.acf.error_correlation: none, processing or a mapping.")
    config = dict(config)
    check_keys(config, {"model", "file", "variance_fraction"}, "Observations.acf.error_correlation")
    fraction = float(config.get("variance_fraction", DEFAULT_VARIANCE_FRACTION))
    if ("model" in config) == ("file" in config):
        raise ValueError("Observations.acf.error_correlation needs exactly one of model, file.")
    if "model" in config:
        if config["model"] != "processing":
            raise ValueError("Observations.acf.error_correlation.model must be 'processing'.")
        blocks = [
            ErrorCorrelation(
                (o.name,),
                processing_error_correlation(o.processing, o.lags[o.mask], o.dt, expected=o.values),
                fraction,
            )
            for o in acfs
        ]
        return blocks, None
    source = path(config)
    errors = read_acf_errors(source)
    components = [o.component for o in acfs]
    missing = [c for c in components if c not in errors.components]
    if missing:
        raise ValueError(f"{source}: no error correlation for components {missing}.")
    for o in acfs:
        if errors.lags.size != o.lags.size or not np.allclose(errors.lags, o.lags):
            raise ValueError(f"{source}: lag grid differs from the {o.name} data file.")
        if not np.allclose(errors.stack[o.component], o.values, rtol=0, atol=1e-6):
            raise ValueError(f"{source}: stack differs from the {o.name} data file.")
    lags = acfs[0].lags[acfs[0].mask]
    if any(not np.array_equal(o.mask, acfs[0].mask) for o in acfs):
        raise ValueError("A joint error correlation needs the same window for all components.")
    block = ErrorCorrelation(tuple(o.name for o in acfs), errors.block(components, lags), fraction)
    return [block], str(source)


def model_parameter_configs(model_cfg: Mapping[str, Any]) -> list[dict]:
    return list((model_cfg.get("Inversion") or {}).get("parameters") or [])
