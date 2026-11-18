# ComplyBay: Overview

ComplyBay models and inverts the data of one ocean-bottom station for a 1-D
seafloor model (layers over a half-space under a water column of known
depth). Three data types are modelled: seafloor compliance, Rayleigh phase
velocity, and teleseismic P-wave autocorrelations (ACF) of the vertical and
pressure channels. The forward solvers, the ACF processing, the model
parameterization with its prior, the likelihood and the sampler are separate
tools, each usable on its own from Python; their inputs may come from other
software. `complybay invert` is one way to combine them for a station, not
the only one.

| tool | page | input | output |
|---|---|---|---|
| `forward.compliance` | [Compliance](compliance.md) | `OBSModel` with water, frequencies (Hz) | normalized compliance $\eta$ (1/Pa) |
| `forward.phase_velocity` | [Rayleigh phase velocity](dispersion.md) | `OBSModel`, periods (s), mode | phase velocity (km/s); NaN where the mode has no root |
| `forward.seafloor_response` | [Teleseismic ACF](acf.md) | `OBSModel`, `ray_p` (s/km), frequencies (Hz) | transfer functions of vertical and radial motion and pressure |
| `forward.synthetic_acfs` | [Teleseismic ACF](acf.md) | `OBSModel`, `ray_p`, `ACFProcessing`, components | lags (s) and one ACF per component, from one response |
| `acf.acf_from_trace`, `acf.stack_errors` | [Teleseismic ACF](acf.md), [Likelihood](likelihood.md) | one P-coda trace and `dt`; per-event ACFs and a lag window | lags and ACF; linear stack, its sigma and the lag correlation |
| `parameterization.parameterization_from_config`, `GeophysicalPrior` | [Parameterization](parameterization.md) | `Model` mapping, parameter vector $\boldsymbol\theta$ | `OBSModel`; log prior, or $-\infty$ with the reason |
| `inference.JointLikelihood` | [Likelihood](likelihood.md) | observations (+ error correlations), `OBSModel`, $\log_{10}$ noise scales | $\ln L$; prediction, $\chi^2$ and $h$ per group, $\chi^2$ and $k$ per block |
| `inference.run_metropolis` | [Sampling](sampling.md) | target with `log_prior_and_likelihood`, start vectors, `MetropolisConfig` | chains of kept draws, acceptance, swap rates |
| `complybay invert` | [Sampling](sampling.md) | `model.yaml`, `obs.yaml` | resolved inputs, `run.log`, chains, model ensemble and posterior predictions (netCDF) |

## Conventions shared by all tools

- **Units** at every interface: km, km/s, g/cm³, s, Hz, s/km. A solver that
  needs other units converts at its own boundary (compliance works in SI).
- **Depth** is km below the seafloor, $z$ positive down, $z = 0$ at the
  seafloor. An `OBSModel` is an optional `WaterLayer` (depth, $v_p$ 1.5 km/s,
  $\rho$ 1.03 g/cm³ by default, no attenuation) over a `LayeredModel` of $n$
  layers and a half-space.
- **Time dependence** $e^{-i\omega t}$ wherever a complex quantity appears
  (seafloor response, complex velocities of [Attenuation](attenuation.md)).
  Compliance and phase velocity are real.
- **Signs.** In the seafloor response $Z$ is vertical ground velocity,
  positive up, and $P$ the pressure above the seafloor, positive in
  compression; the ACF uses only $\lvert H\rvert$, so the sign of a component
  does not change it. Compliance takes $u_z$ positive down, so $\eta > 0$.
- **Compliance normalization:** $\eta = k\,u_z/p$, with $k$ (1/m) the
  wavenumber of the gravity wave in the water; a uniform half-space then
  gives a frequency-independent $\eta$ in the static limit.
- **Attenuation.** With `qp`, `qs` set, `vp` and `vs` are phase velocities at
  `q_reference_frequency` (1 Hz by default, assumed); every solver disperses
  them to its own frequencies.

## Quick start

```bash
cd examples/synthetic
complybay invert --model model.yaml --obs obs.yaml -o run
```

`examples/synthetic/` holds a complete station (synthetic data made by
`tools/make_synthetic_example.py`; per-event ACFs in `obs_events.yaml`).
The keys each data type needs are on its page; all keys and defaults are in
the API reference (`complybay.workflow.config`).

## Building these pages

`pip install -e ".[docs]"`, then `mkdocs serve` to preview or
`mkdocs build --strict` to check. Each figure and formula snippet is made by
`python docs/scripts/<name>.py` (package installed), which writes
`docs/img/<name>.png` or `docs/generated/*.md` and asserts its numbers.
