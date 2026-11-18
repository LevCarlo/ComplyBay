"""Table attenuation_effect.md: what sediment Q changes in the data of the synthetic example.

The true model of ``examples/synthetic`` (``TRUTH`` and ``FINE`` layering of
``tools/make_synthetic_example.py``: 2 km of water, 0.6 km of sediment with Vs
0.15-0.6 km/s and Vp 1.75 km/s over a fixed crust and mantle) is made
attenuating in the sediment only. For each (Qs, Qp, f_ref) the script
computes the four data types of the example on their own grids and compares
them with the elastic model, in units of the example's data sigma
(compliance 10 %, phase velocity 0.02 km/s, ACF 0.05 per lag over 0.3-8 s).
Forward calculations only.

Run: python docs/scripts/attenuation_effect.py
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
from _style import GENERATED_DIR

from complybay.acf import preset
from complybay.forward import compliance, phase_velocity, synthetic_acfs
from complybay.io import load_yaml
from complybay.model import OBSModel
from complybay.parameterization import discretize
from complybay.workflow.config import water_from_station
from complybay.workflow.setup import build_inversion

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "synthetic"
_spec = importlib.util.spec_from_file_location(
    "make_synthetic_example", ROOT / "tools" / "make_synthetic_example.py"
)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)

# Data grids and sigma of tools/make_synthetic_example.py and examples/synthetic/obs.yaml.
FREQS = np.geomspace(0.004, 0.025, 15)  # Hz
PERIODS = np.linspace(8.0, 30.0, 12)  # s
RAY_P, WINDOW, ACF_SIGMA = 0.06, (0.3, 8.0), 0.05
DISPERSION_SIGMA, COMPLIANCE_REL_SIGMA = 0.02, 0.1
CASES = [  # (Qs, Qp, f_ref in Hz)
    (10, np.inf, 1.0),
    (20, np.inf, 1.0),
    (50, np.inf, 1.0),
    (20, 100, 1.0),
    (20, 30, 1.0),
    (20, 100, 0.1),
    (20, 100, 0.01),
]

setup = build_inversion(load_yaml(EXAMPLE / "model.yaml"))
p = setup.parameterization
solid = discretize(p.vector_to_model([_example.TRUTH[n] for n in p.names]), _example.FINE)
elastic = OBSModel(solid, water_from_station(load_yaml(EXAMPLE / "obs.yaml")["Station"]))
sediment = solid.tops[:-1] < _example.TRUTH["basement.depth"] - 1e-9
assert elastic.water.thickness == 2.0 and sediment.sum() == 60


def data(model):
    lags, acfs = synthetic_acfs(model, RAY_P, preset("kim2023"))
    keep = (lags >= WINDOW[0]) & (lags <= WINDOW[1])
    return {
        "compliance": compliance(model, FREQS),
        "dispersion": phase_velocity(model, PERIODS),
        "ACF Z": acfs["vertical"][keep],
        "ACF P": acfs["pressure"][keep],
    }


reference = data(elastic)
sigma = {
    "compliance": COMPLIANCE_REL_SIGMA * reference["compliance"],
    "dispersion": DISPERSION_SIGMA,
    "ACF Z": ACF_SIGMA,
    "ACF P": ACF_SIGMA,
}
assert np.all(np.isfinite(reference["dispersion"]))

rows = []
for qs, qp, f_ref in CASES:

    def q(value):
        return np.r_[np.where(sediment, value, np.inf), np.inf]

    attenuating = elastic.with_solid(
        replace(solid, qs=q(qs), qp=q(qp), q_reference_frequency=f_ref)
    )
    cells = []
    for name, values in data(attenuating).items():
        z = (values - reference[name]) / sigma[name]
        cells.append(f"{np.mean(z**2):.3f} ({np.max(np.abs(z)):.2f})")
    qp_text = "inf" if np.isinf(qp) else f"{qp:g}"
    rows.append(f"| {qs:g} | {qp_text} | {f_ref:g} | " + " | ".join(cells) + " |")

table = (
    "| $Q_s$ | $Q_p$ | $f_\\mathrm{ref}$ (Hz) | compliance | dispersion | ACF Z | ACF P |\n"
    "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
)
GENERATED_DIR.mkdir(exist_ok=True)
(GENERATED_DIR / "attenuation_effect.md").write_text(table, encoding="utf-8")
print(table)
