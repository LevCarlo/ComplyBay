"""Inversion results as xarray datasets (written as netCDF4 with h5netcdf).

Files of a run (stage ``prior`` or ``posterior``), variables with their
dimensions:

``<stage>_mcmc.nc`` (:func:`mcmc_dataset`); draws are the kept (post burn-in,
thinned) states of the stored replica (T = 1 by default; the first of the
ladder with tempering):

- ``theta`` (chain, draw, parameter); ``log_prior``, ``log_likelihood``
  (chain, draw);
- ``accepted`` (chain, step), every step including burn-in;
  ``acceptance_rate`` (chain), after burn-in;
- ``proposal_sigma``, ``initial_theta`` (chain, parameter); ``rhat``, ``ess``
  (parameter), split R-hat and effective sample size;
- ``trace_log_posterior`` (chain, trace_step): every ``thin`` steps from step
  0, burn-in included;
- posterior only, the per-group misfits: ``group_chi2`` (chain, draw, group),
  chi2 with the stated sigma (h = 1; lags taken as independent for ACF groups
  in a correlated block); ``group_n`` (group), the fitted samples N;
  ``term_log_likelihood`` (chain, draw, term), the additive log-likelihood
  terms (independent groups, then correlated blocks; they sum to
  ``log_likelihood``); ``trace_group_chi2`` (chain, trace_step, group) and
  ``trace_term_log_likelihood`` (chain, trace_step, term);
- with ``update="covariance"``: ``proposal_covariance`` (chain, parameter,
  parameter_2), the frozen proposal;
- with tempering: ``swap_acceptance`` (chain, temperature_pair);
- attributes: the sampler settings (``n_steps``, ``burn_in``, ``thin``,
  ``update``, ``covariance_start``, ``temperature``, ``temperatures``,
  ``swap_interval``, ``global_jump_interval``), ``stage``, ``lagging_chains``.

``<stage>_model.nc`` (:func:`model_dataset`), for up to
``Output.n_model_samples`` evenly spaced draws:

- ``vs``, ``vp``, ``rho``, ``vp_vs`` (sample, depth) and ``<name>_quantile``
  (quantile, depth) at the quantiles 0.05, 0.16, 0.5, 0.84, 0.95;
- per sample (:func:`complybay.inference.derived_quantities`):
  ``depth.<interface>``, ``width.<interface>``, ``thickness.<segment>``,
  ``tp2.<segment>``, ``ts2.<segment>`` (vertical two-way P and S times),
  ``vs_mean.<segment>``, ``vs_sqrt_mean.<segment>``, ``vs_mid.<segment>``.

``posterior_predictive.nc`` (:func:`predictive_dataset`), per data group
``<g>`` on its axis ``<g>_frequency``, ``<g>_period`` or ``<g>_lag``:

- ``<g>_observed``, ``<g>_sigma``, ``<g>_fitted`` (1 where the sample enters
  the likelihood);
- ``<g>_predicted`` (predictive_sample, axis), predictions of
  ``Output.n_predictive_samples`` random draws; ``<g>_best``, the prediction
  of the draw with the highest log posterior;
- attributes ``<g>_best_chi2``, ``<g>_best_n``, ``<g>_best_noise_scale``,
  ``block_<name>_best_chi2`` and ``block_<name>_k`` per correlated block, and
  ``best_log_posterior``.

``acf_errors.nc`` (:func:`write_acf_errors`, :func:`read_acf_errors`), the
input of ``error_correlation: {file: ...}`` in ``obs.yaml``: ``stack``,
``sigma`` (component, lag) and ``correlation`` (component, window_lag,
component_2, window_lag_2); attributes ``n_events``, ``shrinkage``,
``shrinkage_target``.

Run metadata (code version, configuration files, seed, ...) is in the global
attributes of every run file. Open with
``xarray.open_dataset(path, engine="h5netcdf")``.
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


def _misfits(result, group_sizes):
    """Per-group chi2 and per-term log L of the kept draws and of the trace."""
    names = list(result.detail_names)
    groups = [n.removeprefix("chi2.") for n in names if n.startswith("chi2.")]
    terms = [n.removeprefix("log_likelihood.") for n in names if n.startswith("log_likelihood.")]
    if len(groups) + len(terms) != len(names):
        raise ValueError(f"Unknown detail names: {names}")
    kept = np.stack([c.details for c in result.chains])
    trace = np.stack([c.trace_details for c in result.chains])
    g = len(groups)
    data = {
        "group_chi2": (("chain", "draw", "group"), kept[..., :g]),
        "term_log_likelihood": (("chain", "draw", "term"), kept[..., g:]),
        "trace_group_chi2": (("chain", "trace_step", "group"), trace[..., :g]),
        "trace_term_log_likelihood": (("chain", "trace_step", "term"), trace[..., g:]),
    }
    if group_sizes is not None:
        data["group_n"] = (("group",), np.array([group_sizes[n] for n in groups]))
    return data, {"group": groups, "term": terms}


def mcmc_dataset(
    result, *, rhat, ess, attrs: Mapping | None = None, group_sizes: Mapping | None = None
) -> xr.Dataset:
    """MCMC samples and diagnostics; ``group_sizes`` (N per data group) adds ``group_n``."""
    names = list(result.names)
    extra, extra_coords = {}, {}
    extra["trace_log_posterior"] = (
        ("chain", "trace_step"),
        np.stack([c.trace_log_posterior for c in result.chains]),
    )
    if result.detail_names:
        misfits, extra_coords = _misfits(result, group_sizes)
        extra.update(misfits)
    if result.chains[0].proposal_covariance is not None:
        extra["proposal_covariance"] = (
            ("chain", "parameter", "parameter_2"),
            np.stack([c.proposal_covariance for c in result.chains]),
        )
    ladder = result.config.ladder
    if len(ladder) > 1:
        extra["swap_acceptance"] = (
            ("chain", "temperature_pair"),
            np.stack([c.swap_acceptance for c in result.chains]),
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
            **({"parameter_2": names} if "proposal_covariance" in extra else {}),
            **({"temperature_pair": np.arange(len(ladder) - 1)} if len(ladder) > 1 else {}),
            "step": np.arange(result.chains[0].accepted.size),
            "trace_step": result.chains[0].trace_steps,
            **extra_coords,
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
                "temperatures": list(ladder) if len(ladder) > 1 else None,
                "swap_interval": result.config.swap_interval if len(ladder) > 1 else None,
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
