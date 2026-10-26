### Fortran extension (`raydep_ft.f95`)

`raydep_ft.f95` is the Fortran implementation of the compliance propagator (from [MDN-Compliance](https://github.com/s-g-mo/MDN-Compliance/blob/main/forward_funcs/raydep_ft.f95)). It is wrapped with f2py and compiled by meson through the `meson-python` build backend (see `meson.build` in this directory); there is no separate build step.

A normal or editable install builds it:

```
pip install .                                  # regular install
pip install --no-build-isolation -e ".[dev]"   # development; rebuilds on import
```

The editable install needs `meson-python`, `meson`, `ninja` and `numpy` in the environment, plus a Fortran compiler (`gfortran`). Keep the Python environment outside the source tree: meson refuses include paths that point into it.
