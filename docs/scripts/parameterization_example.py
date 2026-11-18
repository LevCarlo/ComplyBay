"""Figure parameterization_example.png: one theta -> continuous model -> layered model.

The model space of examples/synthetic/model.yaml (sediment Vs gradient with its
own constant Vp and Gardner density over a Brocher crust and a mantle with
fixed Vp/Vs), with the basement made gradational (sigmoid, sampled width) to
show the blend. theta = the example's true values plus a 0.15 km width.

(a) Vs and Vp: continuous model (black), the two neighbours' own profiles
inside the transition support (dashed), and the layers the solvers see (blue,
the example's ``layers_per_segment`` and the default 16 transition layers).
(b) Density, same layout. (c) Error of the vertical two-way S time through
the sediment (hard basement) against the number of sediment layers; dashed:
slope n^-2 (midpoint rule). Known answer: 2 H ln(b / t) / (b - t) for Vs
rising linearly from t to b over H (asserted against ``layering_report``).
"""

import copy
from pathlib import Path

import _style
import numpy as np
from _style import plt

from complybay.io import load_yaml
from complybay.parameterization import (
    DiscretizationConfig,
    discretize,
    layering_report,
    parameterization_from_config,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "synthetic"
config = load_yaml(EXAMPLE / "model.yaml")
model_cfg = copy.deepcopy(config["Model"])
model_cfg["water"] = {"depth_km": 2.0}  # obs.yaml Station.water of the example
model_cfg["interfaces"][0]["width_km"] = {"parameter": "basement.width"}
parameters = list(config["Inversion"]["parameters"]) + [
    {
        "name": "basement.width",
        "initial": 0.0,
        "prior": {"type": "half_normal", "scale": 0.05, "bounds": [0.0, 0.2]},
    }
]
p = parameterization_from_config({"Model": model_cfg, "Inversion": {"parameters": parameters}})
truth = {"basement.depth": 0.6, "sed.vs_top": 0.15, "sed.vs_bottom": 0.6, "sed.vp": 1.75}
theta = np.array([truth.get(name, 0.15) for name in p.names])  # width 0.15 km
model = p.vector_to_model(theta)
layers = DiscretizationConfig(layers_per_segment=config["Discretization"]["layers_per_segment"])
layered = discretize(model, layers)
print("theta:", dict(zip(p.names, theta, strict=True)))

z = np.linspace(0.0, 1.4, 1401)
vs, vp, rho = model.evaluate_properties(z)
top, bottom = model.interfaces[0].support_km
inside = (z > top) & (z < bottom)
upper = model.segments[0].properties(z[inside], model.segment_scaling(0))
lower = model.segments[1].properties(z[inside], model.segment_scaling(1))
edges = np.r_[layered.tops[:-1], layered.tops[1:]].reshape(2, -1).T


def staircase(ax, values, **kwargs):
    """Layer values as steps: (v_i, top_i) -> (v_i, bottom_i) -> (v_i+1, top_i+1) ..."""
    keep = edges[:, 0] < z[-1]
    depth = np.minimum(edges[keep], z[-1]).ravel()
    ax.plot(np.repeat(values[:-1][keep], 2), depth, **kwargs)


fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.3, 0.8, 1.3]}
)
for ax, items in (
    (
        ax_a,
        ((vs, upper[0], lower[0], layered.vs, "Vs"), (vp, upper[1], lower[1], layered.vp, "Vp")),
    ),
    (ax_b, ((rho, upper[2], lower[2], layered.rho, "rho"),)),
):
    for k, (cont, up, low, lay, name) in enumerate(items):
        ax.plot(cont, z, color=_style.TRUTH, lw=1.3, label="continuous" if k == 0 else None)
        ax.plot(
            up,
            z[inside],
            color=_style.APPROX,
            ls="--",
            lw=0.9,
            label="neighbours" if k == 0 else None,
        )
        ax.plot(low, z[inside], color=_style.APPROX, ls="--", lw=0.9)
        staircase(ax, lay, color=_style.PACKAGE, lw=1.0, label="layers" if k == 0 else None)
        ax.text(
            cont[-1] + (0.1 if ax is ax_a else -0.02),
            1.33,
            name,
            fontsize=8,
            ha="left" if ax is ax_a else "right",
        )
    ax.axhspan(top, bottom, color=_style.SECOND, alpha=0.12, lw=0)
    ax.set_ylim(z[-1], 0)
    ax.set_ylabel("depth below seafloor (km)")
ax_a.set_xlabel("velocity (km/s)")
ax_a.legend(loc="lower left")
ax_b.set_xlabel(r"density (g/cm$^3$)")
_style.panel_label(ax_a, "(a)")
_style.panel_label(ax_b, "(b)")

# (c) Two-way vertical S time error of the sediment layering (half-space at the basement).
hard = p.vector_to_model(np.where(np.array(p.names) == "basement.width", 0.0, theta))
counts = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48])
t, b, h = truth["sed.vs_top"], truth["sed.vs_bottom"], truth["basement.depth"]
ts2_exact = 2 * h * np.log(b / t) / (b - t)  # 2 * integral of dz / Vs for a linear Vs
errors = []
for n in counts:
    lay = discretize(hard, DiscretizationConfig(layers_per_segment={"sediment": int(n)}, zmax=h))
    report = layering_report(hard, lay)
    ts2_layers = 2 * np.sum(lay.thickness / lay.vs[:-1])
    eps = report.vs_vertical_time_relative_error
    assert np.isclose(ts2_layers, ts2_exact * (1 + eps), rtol=1e-10)  # report = analytic error
    assert report.vp_vertical_time_relative_error == 0.0  # constant sediment Vp: exact
    errors.append(abs(ts2_layers - ts2_exact) * 1e3)  # ms
errors = np.array(errors)
ax_c.loglog(counts, errors, "o-", ms=3, color=_style.PACKAGE, label="Vs (two-way)")
ax_c.loglog(
    counts,
    errors[0] * (counts / counts[0]) ** -2.0,
    color=_style.APPROX,
    ls="--",
    lw=0.8,
    label=r"$\propto n^{-2}$",
)
# Midpoint rule on a smooth 1 / Vs: second order, so the error falls ~ n^-2.
slope = np.polyfit(np.log(counts[3:]), np.log(errors[3:]), 1)[0]
assert -2.1 < slope < -1.9, slope
ax_c.axvline(12, color=_style.APPROX, lw=0.8, ls=":")
ax_c.set_xlabel("layers in the sediment")
ax_c.set_ylabel("|two-way time error| (ms)")
ax_c.legend(loc="upper right")
_style.panel_label(ax_c, "(c)")
fig.tight_layout()
_style.save(fig, __file__)
print(
    f"ts2 = {ts2_exact:.4f} s; error with 12 sediment layers {errors[counts == 12][0]:.2f} ms "
    f"(relative {errors[counts == 12][0] / 1e3 / ts2_exact:.1e})"
)
