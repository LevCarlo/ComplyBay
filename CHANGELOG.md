# Changelog

User-visible changes, newest first. **Changed results** lists changes that
alter numbers an existing configuration or call produces, and names the
option that restores the old behavior where one exists.

## Unreleased (branch `dev`, version 0.2.0.dev0)

### Changed results

- **Water-wave wavenumber in the new compliance API.**
  `complybay.forward.compliance` solves `omega^2 = g k tanh(k H)` exactly by
  default (`wavenumber="exact"`). v0.1 used the series approximation `poly`,
  which is up to ~2 % off in k near k H ~ 1-2 and up to ~3 % off in
  compliance. The legacy functions (`ncomp_fortran_batch`, `k_method="poly"`)
  are unchanged; the old numbers come back with `wavenumber="poly"`. When the
  inversion moves to the new API, compliance predictions change by this
  amount.
- **Compliance on deep models.** The new `compliance` cuts the model, per
  frequency, below the depth where the loading has decayed by e^-25
  (`decay_cutoff=25`). Models of a few km (all v0.1 cases) are not affected,
  and the legacy baseline is reproduced; for deep models the propagator used
  to overflow and return NaN at higher frequencies, and now returns finite
  values. A larger `decay_cutoff` keeps more of the model.
- **Fortran build.** The propagator is compiled by meson in release mode
  instead of plain f2py; results move by up to ~4e-9 (relative). No option
  restores the old build.

### Added

- `complybay.model`: `WaterLayer`, `LayeredModel`, `OBSModel`, one model
  (km, km/s, g/cm^3, depth from the seafloor) shared by all solvers.
- `complybay.forward`:
  - `compliance`, `compliance_kernel`, `water_wavenumber`, `halfspace_compliance`;
  - `phase_velocity`, `phase_velocity_kernel` (disba, water as a fluid top layer);
  - `seafloor_response` (plane-P-wave vertical, radial and pressure response
    under a water column) and `synthetic_acf`.
- `complybay.acf`: `ACFProcessing` shared by observed traces
  (`acf_from_trace`) and synthetics, chained zero-phase filters (`Bandpass`,
  `Highpass`, `Lowpass`, `GaussianFilter`), linear and phase-weighted
  stacking, presets `tauzin2019` (unverified starting values) and `kim2023`
  (smoothing-width unit, water level and high-pass order assumed).
- `synthetic_acfs` (several components from one response),
  `wraparound_fraction` (checks that `duration` is long enough), and the
  `snr` option that adds a flat noise level as in Kim et al. (2023) Text S4.
- `complybay.parameterization` (adapted from SeisForge `seisforge.inv`,
  commit 019d568): segments with constant/gradient/B-spline Vs profiles and an
  optional sampled Vp/Vs profile, interfaces with sampled depths and sigmoid or
  smootherstep widths, per-segment Vp/density relations (density from Vp),
  interface-aware layering with `layering_report`, uniform/normal/half-normal
  priors, and the theta -> `OBSModel` mapping with a fixed water column
  (`docs/parameterization.md`). Segments may carry their own sampled `vp`
  profile (used for sediment) or a `vp_vs` profile.
- `GeophysicalPrior` (log prior = -inf with a reason for out-of-bounds,
  invalid-model and constraint failures), hard constraints that can be limited
  to named segments (`vp_vs_range`, `vs_range`, `monotonic_vs`,
  `boundary_non_decreasing_vs`, `max_vs_gradient`, `vp_above_water`), and
  `sample_prior` for exact independent prior samples with rejection
  diagnostics.
- `complybay.inference`: compliance, dispersion and ACF observations, a joint
  Gaussian likelihood with one sampled noise scale (`log10 h`) per data group,
  Z and P ACFs predicted from one shared response, and `PosteriorTarget` with
  a prior-only mode (`docs/inference.md`).
- Metropolis sampler (`run_metropolis`, adapted from SeisForge): one- or
  all-parameter Gaussian updates, burn-in-only proposal adaptation, global
  uniform jumps (one parameter at a time in one-parameter mode), a temperature setting as the parallel-tempering hook, and
  serial/thread/process chains with reproducible seeds.
- Inversion workflow (`complybay invert`, `complybay.workflow`): `model.yaml`
  (model space, priors, constraints, layering, sampler, output) and `obs.yaml`
  (station with water depth, three-column data files, per-group noise scales,
  forward settings), validated with explicit errors for unknown keys; run
  directories with resolved configs plus run metadata, a log with R-hat/ESS and
  per-group fit, and netCDF files of the chains, the model ensemble with
  derived quantities, and the posterior predictive (`docs/workflow.md`).
  Example with synthetic data in `examples/synthetic/`
  (`tools/make_synthetic_example.py`). New dependencies: xarray, h5netcdf, h5py.
- Convergence diagnostics: split R-hat and multi-chain effective sample size.
- `update="covariance"` (`Sampler.update: covariance`): adaptive Metropolis
  with a covariance learned during burn-in and frozen afterwards
  (`covariance_start`); the frozen proposal covariance is saved in
  `<stage>_mcmc.nc`. `examples/synthetic/model.yaml` now uses it (R-hat
  <= 1.014 and ESS 490-2300 instead of 1.09-1.12 and 40-55 with `one`); the
  `Sampler.update` default stays `one`.
- Regression baseline of the v0.1 forward outputs
  (`tests/data/legacy_v0_1_forward.npz`, `tools/make_legacy_baseline.py`).
- Tests against analytic and independent solutions; `docs/forward.md`,
  `docs/development.md`, `CLAUDE.md`; GitHub Actions CI; MIT `LICENSE`.

### Changed

- ACF forward 4-8x faster (32-layer model: Z + P 34 ms -> 4 ms with `kim2023`,
  8 ms with `tauzin2019`): one response shared by all components
  (`synthetic_acfs`), frequencies above `ACFProcessing.frequency_limit`
  dropped, and the propagator folded into one matrix product per layer.
  Dropping those frequencies (`band_tolerance=1e-6`) changes ACFs by less than
  1e-5; `band_tolerance=0` keeps every frequency. The whitening water level now
  refers to the maximum smoothed amplitude within that band.
- `kim2023` preset follows Kim et al. (2023) Supporting Text S2: whitening by
  the running mean of the power spectrum (`Whitening(average="power")`) and no
  water level, instead of the amplitude average and 0.01 water level assumed
  before. The whitening window now has 2N + 1 samples for width W = 2N df for
  both averages (it had round(W / df) samples), which shifts `tauzin2019`
  ACFs slightly.
- Import name is now `complybay` (was `ComplyBay`), in a `src/` layout.
- Build uses `meson-python`; `setup.py` and `numpy.distutils` are gone, and
  the package installs on Python 3.10-3.13 with NumPy 2.
- Example data moved from `ComplyBay/tests/Hydrate_Ridge` to
  `examples/Hydrate_Ridge`.

### Fixed

- `ComplyKernel` failed under NumPy 2 (stored a size-1 array in a scalar
  element).
- `run_comply_inv` / `run_joint_inv` failed under NumPy 2 when reading the
  water depth.

### Known issues

- Legacy `inversion.inv.forward_sw` calls `RhoEstimate` without Vp, so
  Vp-based density relations fail there; it will be replaced together with
  the inversion layer.
- No solver includes attenuation (Q).
