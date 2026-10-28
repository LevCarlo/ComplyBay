# ComplyBay

**ComplyBay** is a modular toolkit for seafloor compliance inversion, built on top of [BayesBay](https://bayes-bay.readthedocs.io/), which supports trans-dimensional Markov Chain Monte Carlo (MCMC) sampling.

## 🔧 Installation

ComplyBay needs Python ≥ 3.10 and a Fortran compiler (`gfortran`); the Fortran
forward solver is compiled during installation by `meson-python`.

```
git clone git@github.com:LevCarlo/ComplyBay.git
cd ComplyBay
pip install .
```

For development (editable install, rebuilt on import; keep the environment
outside the source tree):

```
pip install meson-python meson ninja numpy
pip install --no-build-isolation -e ".[dev]"
pytest
ruff check .
```

## 🚀 Usage

Invert one station from a model-space file and an observation file:

```
cd examples/synthetic
complybay invert --model model.yaml --obs obs.yaml -o run_posterior
complybay invert --model model.yaml -o run_prior --prior-only
```

See [docs/workflow.md](docs/workflow.md) for the file formats and outputs.
The v0.1 scripts (`run_comply_inv`, `run_joint_inv`, example in
`examples/Hydrate_Ridge`) still work but will be removed.

## 🛠 Development

- [docs/development.md](docs/development.md): code layout, units, testing and
  reproducibility rules for contributions.
- [docs/forward.md](docs/forward.md): forward solvers, conventions and how they
  were validated.
- [docs/parameterization.md](docs/parameterization.md): model parameterization
  for the inversion (segments, interfaces, layering, priors).
- [docs/inference.md](docs/inference.md): observations, likelihood with noise
  scales, and the posterior target.
- [docs/workflow.md](docs/workflow.md): configuration files, running an
  inversion, and the output files.
- [docs/synthetic_tests.md](docs/synthetic_tests.md): what each data type
  constrains, from inversions of a known model.
- [CHANGELOG.md](CHANGELOG.md): changes, including those that alter results.

## 📚 Resources

* What is seafloor compliance? — Prof. **Wayne Crawford** [homepage]( https://www.ipgp.fr/~crawford/compliance.html)
* [MDN-Compliance](https://github.com/s-g-mo/MDN-Compliance): Seafloor compliance inversion using mixture density networks (MDNs).
* [OBStools](https://github.com/nfsi-canada/OBStools): Tools for processing ocean-bottom seismic (OBS) data and measuring seafloor compliance.

## 👤 License

This project is licensed under the MIT License.

It uses [BayesBay](https://github.com/robfitzsimons/bayesbay), which is distributed under the BSD-2-Clause License.

## 📖 Citation

If you use **ComplyBay** in your research, please consider citing the following:

* ... [Hydrate Ridge] 🙏🏻🙏🏻🙏🏻

* Zheng, M., Sheehan, A. F., Liu, C., Wu, M., & Ritzwoller, M. H. (2024). Characterizing Sub-Seafloor Seismic Structure of the Alaska Peninsula Along the Alaska-Aleutian Subduction Zone. *Journal of Geophysical Research: Solid Earth*, *129*(11), e2024JB029862. https://doi.org/10.1029/2024JB029862
* Magrini, F., He, J., & Sambridge, M. (2025). BayesBay: A Versatile Bayesian Inversion Framework Written in Python. *Seismological Research Letters*, *96*(3), 2052–2064. https://doi.org/10.1785/0220240275