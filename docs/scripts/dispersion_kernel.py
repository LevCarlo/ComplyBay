"""Figure dispersion_kernel.png: depth extent of Rayleigh Vs kernels; where to put the model bottom.

An assumed oceanic model (below) under 4 km of water, in layers of at most
2 km. ``phase_velocity_kernel`` returns dc/dVs per layer, roughly the integral
of the sensitivity over the layer, so (a) shows it divided by the layer
thickness (per km, each period scaled by its maximum) and (b) sums it.
(a) Kernels for several periods. (b) Fraction of the summed |kernel|
above each depth; dots mark the 90 % and 99 % depths, which are also written
to docs/generated/dispersion_kernel_depths.md. The half-space starts at
400 km; the script checks that it holds less than 0.1 % of the summed |kernel|
at every period shown, so the depths are not cut by the model bottom.

Run: python docs/scripts/dispersion_kernel.py
"""

from __future__ import annotations

import _style
import numpy as np
from _style import GENERATED_DIR, plt

from complybay.forward.dispersion import phase_velocity_kernel
from complybay.model import LayeredModel, OBSModel, WaterLayer

# Assumed oceanic model (not from a source): sediment, crust, lid, low-velocity zone,
# deeper mantle; tops at 0, 0.4, 7, 80, 200 km, half-space from 400 km.
SOLID = LayeredModel(
    thickness=[0.4, 6.6, 73.0, 120.0, 200.0],
    vp=[1.8, 6.5, 8.1, 7.9, 8.4, 8.9],
    vs=[0.3, 3.7, 4.55, 4.3, 4.6, 4.9],
    rho=[1.8, 2.9, 3.35, 3.35, 3.45, 3.6],
)
MODEL = OBSModel(SOLID.resampled(2.0), WaterLayer(4.0))
PERIODS = (15.0, 30.0, 60.0)

tops = MODEL.solid.tops
mid = np.append(tops[:-1] + 0.5 * MODEL.solid.thickness, tops[-1])
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
rows = []
for period, color in zip(PERIODS, _style.SERIES, strict=False):
    kernel = phase_velocity_kernel(MODEL, period, "vs")
    weight = np.abs(kernel)
    assert weight[-1] < 1e-3 * weight.sum(), (period, weight[-1] / weight.sum())
    cumulative = np.cumsum(weight) / weight.sum()
    # Depth at which the cumulative fraction is reached, interpolated within the layer.
    bottoms = np.append(tops[1:], tops[-1])
    z90, z99 = (np.interp(level, cumulative, bottoms) for level in (0.9, 0.99))
    per_km = kernel[:-1] / MODEL.solid.thickness
    peak = mid[np.argmax(np.abs(per_km))]
    rows.append(f"| {period:g} | {peak:.0f} | {z90:.0f} | {z99:.0f} |")
    scale = np.abs(per_km).max()
    ax_a.plot(per_km / scale, mid[:-1], color=color, lw=1.2, label=f"{period:g} s")
    ax_b.plot(cumulative[:-1], bottoms[:-1], color=color, lw=1.2)
    ax_b.plot([0.9, 0.99], [z90, z99], "o", ms=4, color=color)
for z in SOLID.tops[1:]:
    ax_a.axhline(z, color="0.6", lw=0.5, ls=":")
    ax_b.axhline(z, color="0.6", lw=0.5, ls=":")
ax_a.invert_yaxis()
ax_a.set_ylim(400, 0)
ax_a.set_xlabel(r"$\partial c/\partial v_s$ per km (scaled)")
ax_a.set_ylabel("depth below seafloor (km)")
ax_a.legend(loc="lower right")
_style.panel_label(ax_a, "(a)")
ax_b.set_xlabel(r"fraction of $\sum|K|$ above depth")
ax_b.set_xlim(0, 1.02)
_style.panel_label(ax_b, "(b)")
fig.tight_layout()
_style.save(fig, __file__)

table = (
    "| period (s) | peak (km) | 90 % above (km) | 99 % above (km) |\n|---|---|---|---|\n"
    + "\n".join(rows)
    + "\n"
)
GENERATED_DIR.mkdir(exist_ok=True)
(GENERATED_DIR / "dispersion_kernel_depths.md").write_text(table, encoding="utf-8")
print(table)
