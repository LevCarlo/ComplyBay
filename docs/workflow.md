# Running an inversion

One station is inverted from two configuration files, following the split
used by SeisForge's dispersion inversion (`examples/STA001`):

| file | contents | changes when |
|---|---|---|
| `model.yaml` | `Model` (segments, interfaces, relations), `Inversion.parameters`, `Constraints`, `Discretization`, `Sampler`, `Output` | you change the model space or sampling |
| `obs.yaml` | `Station` (name, water depth), `Observations` (data files and their settings, noise scales), `Forward` (solver settings) | you change station or data |

The water column is a station property, so it is in `obs.yaml` and one
`model.yaml` serves stations at different depths. A complete example with
synthetic data is in `examples/synthetic/` (data made by
`tools/make_synthetic_example.py`).

```bash
cd examples/synthetic
complybay invert --model model.yaml --obs obs.yaml -o run_posterior
complybay invert --model model.yaml -o run_prior --prior-only     # no data needed
complybay invert --model model.yaml --obs obs.yaml -o run_both --with-prior
```

Options: `--overwrite` (write into an existing directory; refused otherwise),
`--progress` / `--progress-every N` (not with the process executor),
`--log-level`. From Python: `complybay.workflow.run_inversion(model_file,
obs_file, output, ...)`, or `build_inversion(...)` to get the setup objects
without running anything.

## Data files

Whitespace-separated, `#` comments, three columns `x  value  sigma`:

| group | x | value |
|---|---|---|
| `compliance` | frequency (Hz) | normalized compliance (1/Pa) |
| `dispersion` | period (s) | phase velocity (km/s) |
| `acf` components | lag (s), starting at 0 with a constant step | stacked ACF |

For the ACF, the sampling step `dt` and `max_lag` come from the lag column,
and the processing (`kim2023`, `tauzin2019`, `{preset: ..., <overrides>}` or a
full mapping of `whitening`, `filters`, `zero_lag_taper`, `normalize`,
`band_tolerance`) must be the one used to make the observed ACF.

## Noise scales

A data group with `noise: {bounds: [-0.5, 0.5]}` gets a sampled parameter
`noise.<group>` = log10 h (group names `compliance`, `dispersion`,
`acf_vertical`, `acf_pressure`); `noise: true` uses the defaults
(bounds [-0.5, 0.5], initial 0, proposal sigma 0.05). Without `noise` the
stated sigma is used as is. See `docs/inference.md` for the likelihood.

## Sampler and output settings

```yaml
Sampler:
  n_steps: 30000          # per chain
  burn_in: 8000
  thin: 10
  n_chains: 4
  update: one             # or all
  adapt: true             # tune proposal sigmas during burn-in only
  global_jump_interval: 20
  executor: process       # serial, thread, process
  seed: 2
  initial: prior          # chains start from exact prior samples; or theta0
Output:
  max_depth_km: 10.0      # depth range of the model ensemble (default: model zmax)
  n_depths: 501
  n_model_samples: 2000
  n_predictive_samples: 200
```

## Output directory

| file | contents |
|---|---|
| `resolved_model.yaml`, `resolved_obs.yaml` | the inputs with a `Run` block: version, git commit, time, files, station, parameter names, seed |
| `run.log` | settings, progress, a parameter table (median, 5-95 %, R-hat, ESS), a warning when R-hat > 1.1, and chi2/N and h of the best sample per data group |
| `<stage>_mcmc.nc` | `theta` (chain, draw, parameter), `log_prior`, `log_likelihood`, `accepted` (chain, step), `acceptance_rate`, adapted `proposal_sigma`, `initial_theta`, `rhat`, `ess` |
| `<stage>_model.nc` | `vs`, `vp`, `rho`, `vp_vs` (sample, depth) and their quantiles (5, 16, 50, 84, 95 %); per sample `depth.<interface>`, `width.<interface>`, `thickness.<segment>`, `tp2.<segment>` / `ts2.<segment>` (vertical two-way P / S time), `vs_mean.<segment>` |
| `posterior_predictive.nc` | per data group: observed values, sigma, fitted mask, predictions of `n_predictive_samples` samples, the best sample's prediction; its chi2, N and h in the attributes |

`<stage>` is `prior` and/or `posterior`. Open with
`xarray.open_dataset(path, engine="h5netcdf")`. The derived quantities are
the robust summaries when profile end values trade off (see
`docs/parameterization.md`).

## The synthetic example as shipped

`complybay invert --model model.yaml --obs obs.yaml -o run` in
`examples/synthetic` takes about 3.7 min (4 processes). All true values lie in
the 90 % intervals, and the noise scales are near 1 (ESS 600-2000), but the
model parameters have R-hat 1.09-1.12 and an ESS of only 40-55 out of 8800
kept draws, so the log warns that the chains are not converged. Strongly
correlated parameters (sediment thickness, Vp and the Vs end values) make the
one-parameter random walk slow; longer chains, or correlated (covariance-
adapted) proposals, are needed for production use.
