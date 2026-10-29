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

For the ACF, `error_correlation` (in `Observations.acf`) sets the error
correlation between lags: `none` (default, independent), `processing`
(theory, per component) or `{file: acf_errors.nc}` (from the event ACFs,
written by `complybay.acf.stack_errors` and `complybay.io.write_acf_errors`;
the stacks in it must equal the data files). Both mapping forms take
`variance_fraction` (default 0.99). See `docs/inference.md`; an example is
`examples/synthetic/obs_events.yaml`.

For the ACF, the sampling step `dt` and `max_lag` come from the lag column,
and the processing (`kim2023`, `tauzin2019`, `{preset: ..., <overrides>}` or a
full mapping of `whitening`, `filters`, `zero_lag_taper`, `normalize`,
`band_tolerance`) must be the one used to make the observed ACF.

### Several ray-parameter groups

Events binned by ray parameter give one stack per bin. `Observations.acf`
then takes a list of groups, each a mapping as above plus a `name` (letters,
digits, `_`):

```yaml
  acf:
    - name: p05
      processing: kim2023
      ray_p: 0.05
      window: [0.3, 8.0]
      error_correlation: {file: acf_errors_p05.nc}
      components:
        vertical: {file: acf_z_p05.dat, noise: {bounds: [-0.5, 1.0]}}
        pressure: {file: acf_p_p05.dat, noise: {bounds: [-0.5, 1.0]}}
    - name: p07
      ray_p: 0.07
      ...
```

Data groups are `acf_<name>_<component>`, each with its own noise scale and
error-correlation block; errors of different bins are taken as independent
(different events). Every group costs one seafloor response per evaluation.

### Setup checks

Before sampling, `run.log` gets one line per check
(`complybay.workflow.checks`; from Python `build_inversion(...).check()`):

| check | level |
|---|---|
| compliance frequencies above the infragravity limit sqrt(g tanh(2 pi) / (2 pi H)), where the wavelength equals the water depth (0.028 Hz in 2 km of water): the wave pressure at the seafloor is attenuated by more than cosh(2 pi) = 268 | warning |
| ACF window beyond the data's last lag | warning |
| ACF window including the normalizing zero lag (`normalize: zero_lag`): fixed to 1, no information, but counted in N | warning |
| water multiples (lags m 2 H sqrt(1/vp_w^2 - p^2)) inside the window, and how far a 10 m water-depth error (assumed) shifts them, in s and in degrees at the passband peak | info |
| no water multiple inside the window: the ACF fit does not depend on the water depth | info |

The water-depth sensitivity matters because the multiples are the strongest
arrivals in the ACF: in the synthetic example (2 km, p = 0.06 s/km, kim2023)
10 m shifts the third multiple at 7.97 s by 0.04 s, 13 degrees at 0.91 Hz.
Use the station depth from a bathymetric or acoustic survey, not a nominal
deployment depth.

## Noise scales

A data group with `noise: {bounds: [-0.5, 0.5]}` gets a sampled parameter
`noise.<group>` = log10 h (group names `compliance`, `dispersion`,
`acf_vertical`, `acf_pressure`, or `acf_<name>_<component>` for grouped ACFs); `noise: true` uses the defaults
(bounds [-0.5, 0.5], initial 0, proposal sigma 0.05). Without `noise` the
stated sigma is used as is. See `docs/inference.md` for the likelihood.

## Sampler and output settings

```yaml
Sampler:
  n_steps: 30000          # per chain
  burn_in: 8000
  thin: 10
  n_chains: 4
  update: one             # or all, or covariance (adaptive Metropolis)
  adapt: true             # tune proposal sigmas during burn-in only
  covariance_start: 4000  # covariance only; default burn_in // 2
  global_jump_interval: 20
  tempering: {n_temperatures: 5, max_temperature: 30}  # parallel tempering; or {temperatures: [...]}; swap_interval: 1
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
| `run.log` | settings, progress, a parameter table (median, 5-95 %, R-hat, ESS), a warning when R-hat > 1.1, a warning per chain stuck below the others (`docs/inference.md`, lagging chains), the tempering ladder and swap acceptance, and chi2/N and h of the best sample per data group |
| `<stage>_mcmc.nc` | `theta` (chain, draw, parameter), `log_prior`, `log_likelihood`, `accepted` (chain, step), `acceptance_rate`, adapted `proposal_sigma`, `initial_theta`, `rhat`, `ess`; with tempering `swap_acceptance` (chain, temperature_pair) and the `temperatures` attribute; attribute `lagging_chains`; per-group misfits (below) |
| `<stage>_model.nc` | `vs`, `vp`, `rho`, `vp_vs` (sample, depth) and their quantiles (5, 16, 50, 84, 95 %); per sample `depth.<interface>`, `width.<interface>`, `thickness.<segment>`, `tp2.<segment>` / `ts2.<segment>` (vertical two-way P / S time), `vs_mean.<segment>` (travel-time average), `vs_sqrt_mean.<segment>`, `vs_mid.<segment>` |
| `posterior_predictive.nc` | per data group: observed values, sigma, fitted mask, predictions of `n_predictive_samples` samples, the best sample's prediction; its chi2, N and h in the attributes |

Per-group misfits in `posterior_mcmc.nc`, because the total log likelihood
mostly follows whichever group changes most and can fall while another group
gets worse:

| variable | dims | meaning |
|---|---|---|
| `group_chi2` | chain, draw, group | chi2 with the stated sigma (h = 1); lags taken as independent for ACF groups in a correlated block |
| `group_n` | group | fitted samples N, for chi2 / N |
| `term_log_likelihood` | chain, draw, term | the additive log-likelihood terms: independent groups, then correlated blocks (sum = `log_likelihood`) |
| `trace_log_posterior`, `trace_group_chi2`, `trace_term_log_likelihood` | chain, trace_step | the same every `thin` steps from step 0, burn-in included |

```python
m = xr.open_dataset("run/posterior_mcmc.nc", engine="h5netcdf")
(m.trace_group_chi2 / m.group_n).sel(chain=0).plot.line(x="trace_step", yscale="log")
```

`<stage>` is `prior` and/or `posterior`. Open with
`xarray.open_dataset(path, engine="h5netcdf")`. The derived quantities are
the robust summaries when profile end values trade off (see
`docs/parameterization.md`).

## The synthetic example as shipped

`complybay invert --model model.yaml --obs obs.yaml -o run` in
`examples/synthetic` (4 model + 4 noise parameters, 4 chains x 30 000 steps,
burn-in 8000, `update: covariance`) takes about 3.3 min in 4 processes. All
true values lie in the 90 % intervals, R-hat is at most 1.014 and the ESS
of the 8800 kept draws is 490-2300 per parameter (lowest for the sediment Vs
end values, which trade off).

With `update: one` and otherwise the same settings, the model parameters had
R-hat 1.09-1.12 and ESS 40-55, and the log warned that the chains were not
converged: sediment thickness, Vp and the Vs end values are strongly
correlated, and one-parameter steps cross that ridge slowly.
