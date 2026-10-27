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
- Regression baseline of the v0.1 forward outputs
  (`tests/data/legacy_v0_1_forward.npz`, `tools/make_legacy_baseline.py`).
- Tests against analytic and independent solutions; `docs/forward.md`,
  `docs/development.md`, `CLAUDE.md`; GitHub Actions CI; MIT `LICENSE`.

### Changed

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
