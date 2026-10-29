"""Record forward-solver outputs of the legacy (v0.1) code as a regression baseline.

Run against a checkout of tag ``v0.1-legacy`` with its Fortran extension built
in place, for example::

    git worktree add /tmp/legacy v0.1-legacy
    cd /tmp/legacy/ComplyBay/forward
    python -m numpy.f2py -c raydep_ft.f95 -m raydep_ft
    cd -
    PYTHONPATH=/tmp/legacy python tools/make_legacy_baseline.py \
        tests/data/legacy_v0_1_forward.npz

The script imports the legacy ``ComplyBay`` package only, so the baseline does
not depend on the refactored code it is used to check. The legacy
``ComplyKernel`` is not recorded: it fails under NumPy 2 (it stores a size-1
array into a scalar element).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ComplyBay.forward import k_solver
from ComplyBay.forward.ncomp_fortran import ncomp_fortran, ncomp_fortran_batch
from ComplyBay.forward.ncomp_matlab_py import calc_ncomp_mat
from ComplyBay.inversion.inv import forward_comply, forward_sw

# Models are (thickness m, rho g/cm^3, vp km/s, vs km/s); the last row is the half-space.
MODELS = {
    "crawford": np.array(
        [
            [1000.0, 2.5, 4.0, 2.5],
            [2000.0, 3.0, 7.0, 4.0],
            [400.0, 2.5, 4.0, 1.0],
            [1000.0, 3.0, 7.0, 4.0],
        ]
    ),
    "sediment_gradient": np.column_stack(
        (
            np.full(20, 50.0),
            np.linspace(1.6, 2.3, 20),
            np.linspace(1.6, 3.2, 20),
            np.linspace(0.15, 1.2, 20),
        )
    ),
    "sediment_over_basement": np.array(
        [
            [300.0, 1.8, 1.7, 0.3],
            [500.0, 2.1, 2.4, 0.8],
            [2000.0, 2.6, 4.5, 2.5],
            [1000.0, 2.9, 6.5, 3.7],
        ]
    ),
}
WATER_DEPTHS = np.array([779.0, 3000.0])
FREQS = np.arange(0.003, 0.0301, 0.001)
K_METHODS = ("poly", "newton", "brent")

# Parameters for the legacy inversion-layer forward functions.
VORONOI_PARAM = {
    "INVERSION": {
        "MODEL": {
            "VORONOI": {"min_depth": 0.0, "max_depth": 1000.0},
            "VP": {"inverse": False, "method": "Castegna1995"},
            "RHO": {"inverse": False, "method": "Brocher2005"},
        }
    }
}
VORONOI_SITES = np.array([25.0, 120.0, 300.0, 550.0, 850.0])
VORONOI_VS = np.array([0.2, 0.45, 0.8, 1.3, 2.0])
SW_PERIODS = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 20.0])


def compliance_cases(out):
    for name, model in MODELS.items():
        for wdepth in WATER_DEPTHS:
            tag = f"{name}__h{int(wdepth)}"
            for method in K_METHODS:
                out[f"ncomp_batch__{tag}__{method}"] = ncomp_fortran_batch(
                    wdepth, FREQS, model.copy(), k_method=method
                )
            out[f"ncomp_batch_log__{tag}"] = ncomp_fortran_batch(
                wdepth, FREQS, model.copy(), log_scale=True
            )
            out[f"ncomp_scalar__{tag}"] = ncomp_fortran(wdepth, FREQS, model.copy())
            out[f"ncomp_python__{tag}"] = calc_ncomp_mat(wdepth, FREQS, model.copy())


def k_solver_cases(out):
    omega = 2 * np.pi * FREQS
    for wdepth in WATER_DEPTHS:
        for method in K_METHODS:
            out[f"k__h{int(wdepth)}__{method}"] = k_solver.solve_k(omega, wdepth, method=method)


def inversion_forward_cases(out):
    state = {"voronoi": {"discretization": VORONOI_SITES, "vs": VORONOI_VS}}
    out["voronoi_sites"] = VORONOI_SITES
    out["voronoi_vs"] = VORONOI_VS
    out["forward_comply"] = forward_comply(state, freqs=FREQS, wdepth=779.0, param=VORONOI_PARAM)
    out["sw_periods"] = SW_PERIODS
    out["forward_sw_rayleigh"] = forward_sw(
        state, VORONOI_PARAM, periods=SW_PERIODS, wave="rayleigh", mode=0, wdepth=779.0
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="output .npz file")
    args = parser.parse_args(argv)

    out = {"freqs": FREQS, "water_depths": WATER_DEPTHS}
    for name, model in MODELS.items():
        out[f"model__{name}"] = model
    compliance_cases(out)
    k_solver_cases(out)
    inversion_forward_cases(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"wrote {len(out)} arrays to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
