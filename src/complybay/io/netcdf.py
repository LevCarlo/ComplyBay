"""Inversion results as xarray datasets (written as netCDF4 with h5netcdf).

Three files per run stage (``prior`` or ``posterior``):

- ``<stage>_mcmc.nc``: theta (chain, draw, parameter), log_prior,
  log_likelihood, accepted (chain, step), acceptance rate, adapted proposal
  sigma, initial theta, split R-hat and effective sample size per parameter;
  with ``update="covariance"`` also the frozen proposal covariance
  (chain, parameter, parameter_2).
- ``<stage>_model.nc``: Vs, Vp, rho, Vp/Vs (sample, depth) for a subset of the
  samples, their quantiles (quantile, depth), and derived quantities per sample
  (interface depths, per-segment thickness, two-way times, mean Vs).
- ``posterior_predictive.nc``: per data group the observed values, sigma,
  fitted mask, predictions of a subset of samples, and the prediction of the
  best (highest posterior) sample with its chi2 and noise scale.

``acf_errors.nc`` (:func:`write_acf_errors`): a stacked ACF with its sigma
and the error correlation between lags and components, the input of
``error_correlation: {file: ...}`` in ``obs.yaml``.

Run metadata (code version, configuration files, seed, ...) is in the global
attributes of every file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from complybay.acf.errors import StackErrors

QUANTILES = (0.05, 0.16, 0.5, 0.84, 0.95)


def _attrs(attrs: Mapping | None) -> dict:
    out = {}
    for key, value in (attrs or {}).items():
        if value is None:
            value = "none"
        elif isinstance(value, bool):
            value = int(value)
        elif isinstance(value, list | tuple):
            value = ", ".join(str(v) for v in value)
        out[key] = value
    return out


def mcmc_dataset(result, *, rhat, ess, attrs: Mapping | None = None) -> xr.Dataset:
    names = list(result.names)
    extra = {}
    if result.chains[0].proposal_covariance is not None:
        extra["proposal_covariance"] = (
            ("chain", "parameter", "parameter_2"),
            np.stack([c.proposal_covariance for c in result.chains]),
        )
    return xr.Dataset(
        data_vars={
            "theta": (("chain", "draw", "parameter"), result.samples),
            "log_prior": (("chain", "draw"), result.log_prior),
            "log_likelihood": (("chain", "draw"), result.log_likelihood),
            "accepted": (
                ("chain", "step"),
                np.stack([c.accepted for c in result.chains]).astype(np.int8),
            ),
            "acceptance_rate": (("chain",), result.acceptance_rates),
            "proposal_sigma": (
                ("chain", "parameter"),
                np.stack([c.proposal_sigma for c in result.chains]),
            ),
            "initial_theta": (
                ("chain", "parameter"),
                np.stack([c.initial_theta for c in result.chains]),
            ),
            "rhat": (("parameter",), np.asarray(rhat, dtype=float)),
            "ess": (("parameter",), np.asarray(ess, dtype=float)),
            **extra,
        },
        coords={
            "chain": np.arange(result.samples.shape[0]),
            "draw": np.arange(result.samples.shape[1]),
            "parameter": names,
            **({"parameter_2": names} if extra else {}),
            "step": np.arange(result.chains[0].accepted.size),
        },
        attrs=_attrs(
            {
                **(attrs or {}),
                "n_steps": result.config.n_steps,
                "burn_in": result.config.burn_in,
                "thin": result.config.thin,
                "update": result.config.update,
                "covariance_start": (
                    result.config.covariance_start_step
                    if result.config.update == "covariance"
                    else None
                ),
                "temperature": result.config.temperature,
                "global_jump_interval": result.config.global_jump_interval,
            }
        ),
    )


def model_dataset(
    depths, vs, vp, rho, derived: Mapping[str, np.ndarray], *, attrs=None
) -> xr.Dataset:
    data = {"vs": vs, "vp": vp, "rho": rho, "vp_vs": vp / vs}
    data_vars = {name: (("sample", "depth"), values) for name, values in data.items()}
    for name, values in data.items():
        data_vars[f"{name}_quantile"] = (
            ("quantile", "depth"),
            np.quantile(values, QUANTILES, axis=0),
        )
    for key, values in derived.items():
        data_vars[key] = (("sample",), values)
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "depth": np.asarray(depths),
            "quantile": list(QUANTILES),
            "sample": np.arange(vs.shape[0]),
        },
        attrs=_attrs(attrs),
    )
    dataset["depth"].attrs.update(units="km", long_name="depth below the seafloor")
    for name, units in (("vs", "km/s"), ("vp", "km/s"), ("rho", "g/cm^3")):
        dataset[name].attrs["units"] = units
    return dataset


def predictive_dataset(
    observations: Sequence,
    predictions: Sequence[np.ndarray],
    best: Sequence[np.ndarray],
    best_fits: Sequence,
    *,
    attrs=None,
) -> xr.Dataset:
    """``predictions[k]``: (n_samples, n_k) for observation k; ``best[k]``: (n_k,)."""
    data_vars, coords, global_attrs = {}, {}, dict(attrs or {})
    for obs, pred, best_pred, fit in zip(observations, predictions, best, best_fits, strict=True):
        name = obs.name
        if hasattr(obs, "freqs"):
            axis, values = f"{name}_frequency", obs.freqs
        elif hasattr(obs, "periods"):
            axis, values = f"{name}_period", obs.periods
        else:
            axis, values = f"{name}_lag", obs.lags
        coords[axis] = values
        mask = obs.mask if hasattr(obs, "mask") else np.ones(values.size, bool)
        data_vars[f"{name}_observed"] = ((axis,), obs.values)
        data_vars[f"{name}_sigma"] = ((axis,), obs.sigma)
        data_vars[f"{name}_fitted"] = ((axis,), mask.astype(np.int8))
        data_vars[f"{name}_predicted"] = (("predictive_sample", axis), np.asarray(pred))
        data_vars[f"{name}_best"] = ((axis,), np.asarray(best_pred))
        global_attrs[f"{name}_best_chi2"] = float(fit.chi2)
        global_attrs[f"{name}_best_n"] = int(fit.n)
        global_attrs[f"{name}_best_noise_scale"] = float(fit.scale)
    coords["predictive_sample"] = np.arange(len(predictions[0]))
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=_attrs(global_attrs))


def write_netcdf(dataset: xr.Dataset, path) -> None:
    dataset.to_netcdf(path, engine="h5netcdf")


def acf_errors_dataset(errors, *, attrs=None) -> xr.Dataset:
    """A :class:`complybay.acf.StackErrors` as a dataset (see :func:`read_acf_errors`)."""
    components = list(errors.components)
    p = errors.window_lags.size
    corr = errors.correlation.reshape(len(components), p, len(components), p)
    return xr.Dataset(
        data_vars={
            "stack": (("component", "lag"), np.stack([errors.stack[c] for c in components])),
            "sigma": (("component", "lag"), np.stack([errors.sigma[c] for c in components])),
            "correlation": (("component", "window_lag", "component_2", "window_lag_2"), corr),
        },
        coords={
            "component": components,
            "component_2": components,
            "lag": errors.lags,
            "window_lag": errors.window_lags,
            "window_lag_2": errors.window_lags,
        },
        attrs=_attrs(
            {
                **dict(errors.attrs),
                **dict(attrs or {}),
                "description": "Linear ACF stack, its sigma and the error correlation between "
                "lags and components (complybay.acf.stack_errors)",
                "n_events": errors.n_events,
                "shrinkage": errors.shrinkage,
                "shrinkage_target": errors.target,
            }
        ),
    )


def write_acf_errors(errors, path, *, attrs=None) -> None:
    write_netcdf(acf_errors_dataset(errors, attrs=attrs), path)


def read_acf_errors(path):
    """Read a file written by :func:`write_acf_errors` back into a ``StackErrors``."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        components = tuple(str(c) for c in ds["component"].values)
        p = ds.sizes["window_lag"]
        n = len(components)
        attrs = dict(ds.attrs)
        return StackErrors(
            lags=ds["lag"].values.copy(),
            components=components,
            stack={c: ds["stack"].sel(component=c).values.copy() for c in components},
            sigma={c: ds["sigma"].sel(component=c).values.copy() for c in components},
            window_lags=ds["window_lag"].values.copy(),
            correlation=ds["correlation"].values.reshape(n * p, n * p).copy(),
            n_events=int(attrs.pop("n_events")),
            shrinkage=float(attrs.pop("shrinkage")),
            target=str(attrs.pop("shrinkage_target")),
            attrs=attrs,
        )
