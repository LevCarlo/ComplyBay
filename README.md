# ComplyBay

**ComplyBay** is a modular toolkit for seafloor compliance inversion, built on top of [BayesBay](https://bayes-bay.readthedocs.io/), which supports trans-dimensional Markov Chain Monte Carlo (MCMC) sampling.

## 🔧 Installation

To install the package and build the Fortran extension:

```
git clone https://github.com/your-username/ComplyBay.git
cd ComplyBay
pip install .
```

For development or editable installs:

```
pip install -e .
```

## 🚀 Usage

To run an inversion using a YAML parameter file:

```
run_complybay_inv -p your_config.yaml -s STATION_NAME
```

Refer to the `/ComplyBay/tests/Hydrate_Ridge` directory for example configurations and test data.

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