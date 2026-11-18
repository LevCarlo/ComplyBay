"""Figure resolution_sensitivity.png: what each data type sees of the sediment, in units of sigma.

Forward calculations only. The true model of examples/synthetic (2 km water,
sediment 0.6 km with Vs 0.15 -> 0.6 km/s and Vp 1.75 km/s, Brocher crust,
fixed mantle), layered as the example's data (60 sediment layers), and the
example's data sampling and sigmas: compliance at 15 frequencies 4-25 mHz
(sigma 10 %), phase velocity at 12 periods 8-30 s (0.02 km/s), Z and P ACF
(kim2023, p = 0.06 s/km, 0.05 per lag, 0.3-8 s). Perturbations: sediment Vs
+20 % at both ends, basement 0.1 km deeper, sediment Vp +0.1 km/s.

Panels: change of the prediction divided by sigma, (a) compliance, (b) phase
velocity, (c) Z-ACF, (d) P-ACF. Also writes docs/generated/resolution_*.md:
chi2 = sum (change / sigma)^2 per group (the drop of 2 ln L at h = 1), the
phase-velocity change at selected periods, and the chi2 of crust Vs top
+0.1 km/s on the two-unit model of examples/synthetic_layered.

Run: python docs/scripts/resolution_sensitivity.py
"""

from pathlib import Path

import _style
import numpy as np
from _style import GENERATED_DIR, plt

from complybay.inference import predict
from complybay.io import load_yaml
from complybay.model import OBSModel
from complybay.parameterization import DiscretizationConfig, discretize
from complybay.workflow import build_inversion

ROOT = Path(__file__).resolve().parents[2] / "examples"
# Truth and data layering of tools/make_synthetic_example.py (TRUTH, FINE).
TRUTH = {"basement.depth": 0.6, "sed.vs_top": 0.15, "sed.vs_bottom": 0.6, "sed.vp": 1.75}
FINE = DiscretizationConfig(layers_per_segment={"sediment": 60, "crust": 40, "mantle": 30})
PERTURBATIONS = {
    "sediment Vs +20 %": {"sed.vs_top": 0.18, "sed.vs_bottom": 0.72},
    "basement 0.1 km deeper": {"basement.depth": 0.7},
    "sediment Vp +0.1 km/s": {"sed.vp": 1.85},
}


def setup_of(directory):
    return build_inversion(
        load_yaml(directory / "model.yaml"), load_yaml(directory / "obs.yaml"), obs_dir=directory
    )


def predictions(setup, values, layers):
    p = setup.parameterization
    theta = [values.get(name, 0.0) for name in p.names]  # noise parameters: log10 h = 0
    return predict(
        setup.target.likelihood.observations,
        OBSModel(discretize(p.vector_to_model(theta), layers), setup.water),
    )


setup = setup_of(ROOT / "synthetic")
observations = setup.target.likelihood.observations
reference = predictions(setup, TRUTH, FINE)
changes = {}
for label, change in PERTURBATIONS.items():
    perturbed = predictions(setup, TRUTH | change, FINE)
    changes[label] = [
        (b - a) / o.sigma for o, a, b in zip(observations, reference, perturbed, strict=True)
    ]


def chi2(obs, normalized):
    mask = getattr(obs, "mask", np.ones(normalized.size, bool))
    return float(np.sum(normalized[mask] ** 2))


# Known answers: a thicker or stiffer sediment lowers the compliance (stiffer seafloor)
# and raises the short-period phase velocity only for Vs; more sediment slows it.
assert np.all(changes["sediment Vs +20 %"][0] < 0)
assert np.all(changes["basement 0.1 km deeper"][0] > 0)
assert changes["basement 0.1 km deeper"][1][0] < 0 < changes["sediment Vs +20 %"][1][0]

fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
x_axes = (
    observations[0].freqs * 1e3,
    observations[1].periods,
    observations[2].lags,
    observations[3].lags,
)
labels = ("frequency (mHz)", "period (s)", "lag (s)", "lag (s)")
titles = ("compliance", "phase velocity", "Z-ACF", "P-ACF")
for k, ax in enumerate(axes):
    for (label, values), color in zip(changes.items(), _style.SERIES, strict=False):
        x, y = x_axes[k], values[k]
        if k >= 2:
            mask = observations[k].mask
            x, y = x[mask], y[mask]
        ax.plot(
            x,
            y,
            color=color,
            lw=1.0 if k >= 2 else 1.4,
            marker=None if k >= 2 else "o",
            ms=3,
            label=label,
        )
    ax.axhline(0, color=_style.TRUTH, lw=0.6)
    ax.set_xlabel(labels[k])
    ax.set_title(titles[k])
    _style.panel_label(ax, f"({'abcd'[k]})")
axes[0].set_ylabel(r"change / $\sigma$")
axes[1].legend(loc="lower right", fontsize=7)
fig.tight_layout()
_style.save(fig, __file__)

# Tables.
lines = ["| perturbation | compliance | phase velocity | Z-ACF | P-ACF |", "|---|---|---|---|---|"]
for label, values in changes.items():
    cells = " | ".join(f"{chi2(o, v):.1f}" for o, v in zip(observations, values, strict=True))
    lines.append(f"| {label} | {cells} |")
layered_dir = ROOT / "synthetic_layered"
truth_cfg = load_yaml(layered_dir / "truth.yaml")
layered_setup = setup_of(layered_dir)
layers = DiscretizationConfig(layers_per_segment=dict(truth_cfg["fine_layers"]))
base = predictions(layered_setup, truth_cfg["parameters"], layers)
crust = predictions(
    layered_setup,
    truth_cfg["parameters"] | {"crust.vs_top": truth_cfg["parameters"]["crust.vs_top"] + 0.1},
    layers,
)
cells = " | ".join(
    f"{chi2(o, (b - a) / o.sigma):.1f}"
    for o, a, b in zip(layered_setup.target.likelihood.observations, base, crust, strict=True)
)
lines.append(f"| crust Vs top +0.1 km/s (two-unit model of `synthetic_layered`) | {cells} |")
(GENERATED_DIR / "resolution_chi2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

periods = observations[1].periods
picks = [int(np.argmin(np.abs(periods - t))) for t in (8, 12, 16, 20, 30)]
table = [
    "| perturbation | " + " | ".join(f"{periods[i]:g} s" for i in picks) + " |",
    "|---|" + "---|" * len(picks),
]
for label, values in changes.items():
    table.append(f"| {label} | " + " | ".join(f"{values[1][i]:.2f}" for i in picks) + " |")
(GENERATED_DIR / "resolution_dispersion.md").write_text("\n".join(table) + "\n", encoding="utf-8")
print("\n".join(lines))
print("\n".join(table))
