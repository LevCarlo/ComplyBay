# ComplyBay

**ComplyBay** does forward modelling and Bayesian inversion for ocean-bottom
seismology. It inverts the data of one ocean-bottom station for a 1-D seafloor
model (sediment, crust, mantle under a water layer of known depth), using
three kinds of data together:

- **seafloor compliance** (infragravity-wave loading);
- **Rayleigh-wave phase velocity** (fundamental mode, with the water as a fluid
  layer);
- **teleseismic P-wave autocorrelations (ACF)** of the vertical and pressure
  channels, processed for data and synthetics by one shared flow (presets
  after Tauzin et al. 2019 and Kim et al. 2023).

The inversion samples the model with Metropolis MCMC: adaptive covariance
proposals, parallel tempering for the multimodal ACF posterior, a sampled
noise scale per data group, and correlated ACF errors from the event ACFs or
from the processing theory. Every forward solver is checked against an
independent solution (`docs/forward.md`).

📖 **Documentation:** <https://levcarlo.github.io/ComplyBay/>

## 🔧 Installation

ComplyBay needs Python ≥ 3.10 and a Fortran compiler (`gfortran`); the Fortran
compliance solver is compiled during installation by `meson-python`.

```bash
git clone git@github.com:LevCarlo/ComplyBay.git
cd ComplyBay
pip install .
```

For development (editable install, rebuilt on import; keep the environment
outside the source tree):

```bash
pip install meson-python meson ninja numpy
pip install --no-build-isolation -e ".[dev]"
pytest
ruff check .
```

## 🚀 Usage

A station is described by two YAML files: `model.yaml` (model space, priors,
sampler) and `obs.yaml` (station, data files, ACF processing, noise model).

```bash
cd examples/synthetic
complybay invert --model model.yaml -o run_prior --prior-only     # look at the prior first
complybay invert --model model.yaml --obs obs.yaml -o run_posterior
```

The output directory holds the resolved inputs, a log with convergence
diagnostics (R-hat, ESS, lagging chains, per-group misfits), the chains, the
model ensemble and the posterior predictions as netCDF. From Python:

```python
from complybay.model import LayeredModel, OBSModel, WaterLayer
from complybay.forward import compliance, phase_velocity, synthetic_acf
from complybay.acf import preset

solid = LayeredModel(thickness=[0.4, 1.5, 5.0], vp=[1.8, 4.5, 6.5, 8.1],
                     vs=[0.3, 2.5, 3.7, 4.55], rho=[1.8, 2.6, 2.9, 3.35])
model = OBSModel(solid, WaterLayer(4.0))            # km, km/s, g/cm^3

eta = compliance(model, [0.005, 0.01, 0.02])       # 1/Pa
c = phase_velocity(model, [8.0, 15.0, 30.0])       # km/s
lags, acf_z = synthetic_acf(model, 0.06, preset("kim2023"), component="vertical")
```

## 📚 Documentation

| page | contents |
|---|---|
| [Running an inversion](https://levcarlo.github.io/ComplyBay/workflow/) | configuration files, setup checks, outputs |
| [Forward solvers](https://levcarlo.github.io/ComplyBay/forward/) | compliance, dispersion, ACF, attenuation; conventions and validation |
| [Model parameterization](https://levcarlo.github.io/ComplyBay/parameterization/) | segments, interfaces, priors, layering |
| [Likelihood and posterior](https://levcarlo.github.io/ComplyBay/inference/) | noise scales, correlated ACF errors, sampler |
| [Synthetic tests](https://levcarlo.github.io/ComplyBay/synthetic_tests/) | what each data type constrains, coverage, stress tests |
| [Development guide](https://levcarlo.github.io/ComplyBay/development/) | code layout, units, testing and reproducibility rules |
| [API reference](https://levcarlo.github.io/ComplyBay/api/model/) | generated from the docstrings |

The same pages are in [`docs/`](docs/); changes, including those that alter
results, are in [CHANGELOG.md](CHANGELOG.md).

**Legacy v0.1.** The original compliance inversion built on
[BayesBay](https://bayes-bay.readthedocs.io/) (`run_comply_inv`,
`run_joint_inv`, example in `examples/Hydrate_Ridge`; tag `v0.1-legacy`) still
works but will be removed; new work uses `complybay invert`.

## 📚 Resources

* What is seafloor compliance? — Prof. **Wayne Crawford** [homepage](https://www.ipgp.fr/~crawford/compliance.html)
* [MDN-Compliance](https://github.com/s-g-mo/MDN-Compliance): Seafloor compliance inversion using mixture density networks (MDNs).
* [OBStools](https://github.com/nfsi-canada/OBStools): Tools for processing ocean-bottom seismic (OBS) data and measuring seafloor compliance.

## 👤 License

This project is licensed under the MIT License.

Rayleigh phase velocities are computed with [disba](https://github.com/keurfonluu/disba)
(BSD-3-Clause). The legacy v0.1 inversion uses
[BayesBay](https://github.com/robfitzsimons/bayesbay) (BSD-2-Clause).

## 📖 Citation

If you use **ComplyBay** in your research, please consider citing the following:

* ... [Hydrate Ridge] 🙏🏻🙏🏻🙏🏻

* Zheng, M., Sheehan, A. F., Liu, C., Wu, M., & Ritzwoller, M. H. (2024). Characterizing Sub-Seafloor Seismic Structure of the Alaska Peninsula Along the Alaska-Aleutian Subduction Zone. *Journal of Geophysical Research: Solid Earth*, *129*(11), e2024JB029862. https://doi.org/10.1029/2024JB029862
* Magrini, F., He, J., & Sambridge, M. (2025). BayesBay: A Versatile Bayesian Inversion Framework Written in Python. *Seismological Research Letters*, *96*(3), 2052–2064. https://doi.org/10.1785/0220240275 (legacy v0.1 inversion)
