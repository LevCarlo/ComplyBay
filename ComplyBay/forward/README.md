### ⚙️ Compiling the Fortran Extension (`raydep_ft.f95`)

The file `ComplyBay/forward/raydep_ft.f95` contains a Fortran implementation of the forward solver (from [MDN-Compliance](https://github.com/s-g-mo/MDN-Compliance/blob/main/forward_funcs/raydep_ft.f95)). It is wrapped via f2py and exposed to Python through the setuptools build process.

#### 🔧 Build Instructions

**Option 1: Recommended (In-place build for development)**

```
python setup.py build_ext --inplace
```

This will generate a shared object .so file (e.g., raydep_ft.cpython-310-x86_64-linux-gnu.so) under `ComplyBay/forward/`, making it directly importable via:

```
from ComplyBay.forward import raydep_ft
```

**Option 2: Clean build using pip (editable mode)**

First, make sure you have the following in `pyproject.toml`:

```
[build-system]
requires = ["setuptools<60", "wheel", "numpy"]
build-backend = "setuptools.build_meta"
```

Then:

```
pip uninstall ComplyBay
pip cache purge
rm -rf build dist ComplyBay.egg-info
find . -name '*.so' -delete
pip install -e .
```

