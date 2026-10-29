# ComplyBay

Forward modelling and Bayesian inversion for ocean-bottom seismology: one
station, a 1-D seafloor model, and three kinds of data — seafloor compliance,
fundamental-mode Rayleigh phase velocity, and teleseismic autocorrelations
(ACF) of the vertical and pressure channels.

Units everywhere: km, km/s, g/cm^3, s, Hz, s/km; depth 0 is the seafloor.

## Quick start

```bash
cd examples/synthetic
complybay invert --model model.yaml --obs obs.yaml -o run_posterior
complybay invert --model model.yaml -o run_prior --prior-only
```

## Pages

- [Running an inversion](workflow.md): configuration files, setup checks,
  sampler settings and the output files.
- [Forward solvers](forward.md): compliance, Rayleigh phase velocity, the ACF
  response and attenuation; conventions and how each was validated.
- [Model parameterization](parameterization.md): segments, profiles,
  interfaces, priors and layering.
- [Likelihood and posterior](inference.md): observations, noise scales,
  correlated ACF errors, and the sampler (covariance proposals, parallel
  tempering).
- [Synthetic tests](synthetic_tests.md): what each data type constrains,
  coverage over noise realizations, and the stress tests.
- [Development guide](development.md): code layout, units, testing and
  reproducibility rules for contributions.

Changes, including those that alter results, are listed in
[`CHANGELOG.md`](https://github.com/LevCarlo/ComplyBay/blob/dev/CHANGELOG.md).

## Building these pages

```bash
pip install -e ".[docs]"   # mkdocs and mkdocs-material
mkdocs serve               # live preview at http://127.0.0.1:8000
mkdocs build               # static site in site/
```
