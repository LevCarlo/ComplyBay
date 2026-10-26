"""Forward solvers must reproduce the legacy (v0.1) outputs.

The baseline is produced by ``tools/make_legacy_baseline.py`` from tag
``v0.1-legacy``; see that script for how to regenerate it.
"""

from pathlib import Path

import numpy as np
import pytest

from complybay.forward import k_solver
from complybay.forward.complykernel import ComplyKernel
from complybay.forward.ncomp_fortran import ncomp_fortran, ncomp_fortran_batch
from complybay.forward.ncomp_matlab_py import calc_ncomp_mat
from complybay.inversion.inv import forward_comply, forward_sw

BASELINE = np.load(Path(__file__).parent / "data" / "legacy_v0_1_forward.npz")
FREQS = BASELINE["freqs"]
WATER_DEPTHS = BASELINE["water_depths"]
MODEL_NAMES = sorted(k.removeprefix("model__") for k in BASELINE.files if k.startswith("model__"))
K_METHODS = ("poly", "newton", "brent")
# The baseline Fortran was built by plain f2py, the package by meson in release
# mode; the different optimisation flags move results by up to ~4e-9.
RTOL = 1e-8

VORONOI_PARAM = {
    "INVERSION": {
        "MODEL": {
            "VORONOI": {"min_depth": 0.0, "max_depth": 1000.0},
            "VP": {"inverse": False, "method": "Castegna1995"},
            "RHO": {"inverse": False, "method": "Brocher2005"},
        }
    }
}


def _model(name):
    return BASELINE[f"model__{name}"].copy()


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("wdepth", WATER_DEPTHS)
@pytest.mark.parametrize("method", K_METHODS)
def test_ncomp_batch(name, wdepth, method):
    tag = f"{name}__h{int(wdepth)}"
    got = ncomp_fortran_batch(wdepth, FREQS, _model(name), k_method=method)
    np.testing.assert_allclose(got, BASELINE[f"ncomp_batch__{tag}__{method}"], rtol=RTOL)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("wdepth", WATER_DEPTHS)
def test_ncomp_scalar_and_log(name, wdepth):
    tag = f"{name}__h{int(wdepth)}"
    np.testing.assert_allclose(
        ncomp_fortran(wdepth, FREQS, _model(name)), BASELINE[f"ncomp_scalar__{tag}"], rtol=RTOL
    )
    np.testing.assert_allclose(
        ncomp_fortran_batch(wdepth, FREQS, _model(name), log_scale=True),
        BASELINE[f"ncomp_batch_log__{tag}"],
        rtol=RTOL,
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("wdepth", WATER_DEPTHS)
def test_python_reference_matches_baseline_and_fortran(name, wdepth):
    tag = f"{name}__h{int(wdepth)}"
    python = calc_ncomp_mat(wdepth, FREQS, _model(name))
    np.testing.assert_allclose(python, BASELINE[f"ncomp_python__{tag}"], rtol=RTOL)
    # The pure-Python port uses the exact water-wave root, as Newton does.
    fortran = ncomp_fortran_batch(wdepth, FREQS, _model(name), k_method="newton")
    np.testing.assert_allclose(fortran, python, rtol=1e-6)


@pytest.mark.parametrize("wdepth", WATER_DEPTHS)
@pytest.mark.parametrize("method", K_METHODS)
def test_k_solver(wdepth, method):
    got = k_solver.solve_k(2 * np.pi * FREQS, wdepth, method=method)
    np.testing.assert_allclose(got, BASELINE[f"k__h{int(wdepth)}__{method}"], rtol=RTOL)


def test_inversion_forward_functions():
    state = {
        "voronoi": {
            "discretization": BASELINE["voronoi_sites"],
            "vs": BASELINE["voronoi_vs"],
        }
    }
    np.testing.assert_allclose(
        forward_comply(state, freqs=FREQS, wdepth=779.0, param=VORONOI_PARAM),
        BASELINE["forward_comply"],
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        forward_sw(
            state, VORONOI_PARAM, periods=BASELINE["sw_periods"], wave="rayleigh", wdepth=779.0
        ),
        BASELINE["forward_sw_rayleigh"],
        rtol=RTOL,
    )


def test_comply_kernel_is_central_difference():
    model = _model("sediment_gradient")
    freq, wdepth, dp, layer = 0.01, 779.0, 0.025, 3
    depth, kernel = ComplyKernel(model.copy(), freq, wdepth, "vs", dp=dp).cal_kernel()

    c0 = ncomp_fortran_batch(wdepth, np.array([freq]), model)[0]
    plus, minus = model.copy(), model.copy()
    plus[layer, 3] *= 1 + dp
    minus[layer, 3] *= 1 - dp
    c_plus = ncomp_fortran_batch(wdepth, np.array([freq]), plus)[0]
    c_minus = ncomp_fortran_batch(wdepth, np.array([freq]), minus)[0]

    np.testing.assert_allclose(depth, np.cumsum(model[:, 0])[:-1])
    np.testing.assert_allclose(kernel[layer], (c_plus - c_minus) / (2 * dp * c0), rtol=1e-12)
