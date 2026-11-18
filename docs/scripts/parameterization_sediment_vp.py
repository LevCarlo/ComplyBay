"""Why the sediment gets its own Vp profile: numbers for docs/parameterization.md.

Writes docs/generated/parameterization_sediment_vp.md. On the true model of
examples/synthetic (2 km water, 0.6 km sediment, Vs 0.15 -> 0.6 km/s, Vp
1.75 km/s), layered as the example's data (60 sediment layers):

1. Vp that relations of Vs give for this sediment, against the water velocity.
2. How much each data group of the example sees the shape of the sediment Vp
   profile: Vp rising 1.52 -> 2.0 km/s, which has the same vertical P time as
   the constant 1.75 km/s (asserted to 0.1 %), against a 5 % thicker sediment.
   Misfit in chi2 with the example's sigmas (compliance 10 %, phase velocity
   0.02 km/s, ACF 0.05 per lag over 0.3-8 s), noise-free.

Run: python docs/scripts/parameterization_sediment_vp.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from _style import GENERATED_DIR

from complybay.inference import derived_quantities, predict
from complybay.io import load_yaml
from complybay.model import OBSModel
from complybay.parameterization import DiscretizationConfig, discretize, model_from_config
from complybay.parameterization.scaling import brocher_vp_from_vs, castagna_vp_from_vs
from complybay.workflow import build_inversion

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "synthetic"
TRUTH = {"basement.depth": 0.6, "sed.vs_top": 0.15, "sed.vs_bottom": 0.6, "sed.vp": 1.75}
FINE = DiscretizationConfig(layers_per_segment={"sediment": 60, "crust": 40, "mantle": 30})

model_yaml = load_yaml(EXAMPLE / "model.yaml")
setup = build_inversion(model_yaml, load_yaml(EXAMPLE / "obs.yaml"), obs_dir=EXAMPLE)
observations = setup.target.likelihood.observations
water = setup.water


def build(model_cfg, values):
    model = model_from_config(model_cfg, values)
    return model, OBSModel(discretize(model, FINE), water)


base_cfg = model_yaml["Model"]
gradient_cfg = copy.deepcopy(base_cfg)
gradient_cfg["segments"][0]["vp"] = {"type": "gradient", "top": 1.52, "bottom": 2.0}
thicker = dict(TRUTH, **{"basement.depth": 0.63})

base_model, base = build(base_cfg, TRUTH)
cases = {
    "Vp 1.52 → 2.0 km/s (same $t_{p2}$)": build(gradient_cfg, TRUTH),
    "sediment 5 % thicker (0.63 km)": build(base_cfg, thicker),
}
tp2 = derived_quantities(base_model)["tp2.sediment"]
assert np.isclose(
    derived_quantities(cases["Vp 1.52 → 2.0 km/s (same $t_{p2}$)"][0])["tp2.sediment"],
    tp2,
    rtol=1e-3,
)

reference = predict(observations, base)
rows = []
for label, (_, obs_model) in cases.items():
    prediction = predict(observations, obs_model)
    chi2, extra = [], {}
    for obs, a, b in zip(observations, reference, prediction, strict=True):
        mask = getattr(obs, "mask", np.ones(a.size, bool))
        chi2.append(float(np.sum(((b - a)[mask] / obs.sigma[mask]) ** 2)))
        if obs.name == "compliance":
            extra["compliance"] = np.max(np.abs(b / a - 1))
        if obs.name == "dispersion":
            extra["dispersion"] = np.max(np.abs(b - a))
    rows.append((label, chi2, extra))

# Vs-based relations at the sediment top and bottom, and a constant Vp/Vs with the same tp2.
ts2 = derived_quantities(base_model)["ts2.sediment"]
ratio = ts2 / tp2  # Vp = r Vs gives tp2 = ts2 / r
vs_ends = np.array([TRUTH["sed.vs_top"], TRUTH["sed.vs_bottom"]])
relations = {
    "`brocher2005`": brocher_vp_from_vs(vs_ends),
    "`castagna1985`": castagna_vp_from_vs(vs_ends),
    f"`constant_vp_vs`, $r$ = {ratio:.2f} (same $t_{{p2}}$)": ratio * vs_ends,
}

lines = [
    "| Vp of the sediment from | at $V_s$ = 0.15 km/s | at $V_s$ = 0.6 km/s |",
    "|---|---|---|",
]
for name, values in relations.items():
    flag = [" (< water)" if v < water.vp else "" for v in values]
    lines.append(f"| {name} | {values[0]:.2f}{flag[0]} | {values[1]:.2f}{flag[1]} |")
lines += [
    "",
    "| change of the sediment | compliance χ² (max \\|Δη/η\\|) | phase velocity χ² "
    "(max \\|Δc\\|) | Z-ACF χ² | P-ACF χ² |",
    "|---|---|---|---|---|",
]
for label, chi2, extra in rows:
    lines.append(
        f"| {label} | {chi2[0]:.3f} ({100 * extra['compliance']:.1f} %) | {chi2[1]:.3f} "
        f"({extra['dispersion']:.4f} km/s) | {chi2[2]:.1f} | {chi2[3]:.1f} |"
    )
GENERATED_DIR.mkdir(exist_ok=True)
(GENERATED_DIR / "parameterization_sediment_vp.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"tp2 = {tp2:.4f} s, ts2 = {ts2:.4f} s")
