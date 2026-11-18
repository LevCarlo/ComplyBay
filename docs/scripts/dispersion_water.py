"""Figure dispersion_water.png: the fundamental Rayleigh mode under water.

(a) Mode-0 phase velocity of the shared docs model (0.6 km of sediment with
Vs 0.3 km/s over crust and mantle) under 0, 1, 2 and 4 km of water. At
short periods every water-loaded curve approaches the Scholte speed of water
over the sediment half-space (closed form of dispersion_derivation.py).
(b) The water-loaded curves relative to the curve without water.

The script also checks that the steep rise near 8 s is mode 0 and not a jump
of disba's root search: separate one-period calls, a ten times finer search
step and a curve started at a longer period give the same values. Mode 1
under 2 km of water (not plotted) is checked to lie above mode 0 and to
depend on dc below about 2.8 s, where its roots lie closer together than the
default step.

Run: python docs/scripts/dispersion_water.py
"""

from __future__ import annotations

import _style
import numpy as np
from _models import SEDIMENT_OVER_CRUST
from _style import plt
from dispersion_derivation import scholte_speed

from complybay.forward.dispersion import phase_velocity
from complybay.model import OBSModel, WaterLayer

DEPTHS_KM = (1.0, 2.0, 4.0)
periods = np.geomspace(0.3, 60.0, 60)
sediment = tuple(SEDIMENT_OVER_CRUST.property(name)[0] for name in ("vp", "vs", "rho"))

dry = phase_velocity(OBSModel(SEDIMENT_OVER_CRUST), periods)
assert np.all(np.isfinite(dry))

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
ax_a.plot(periods, dry, color=_style.TRUTH, lw=1.2, label="no water")
for depth_km, color in zip(DEPTHS_KM, _style.SERIES, strict=False):
    model = OBSModel(SEDIMENT_OVER_CRUST, WaterLayer(depth_km))
    wet = phase_velocity(model, periods)
    assert np.all(np.isfinite(wet))
    # Shortest period: within 1 % of the Scholte speed (the 0.3 s wave decays within a
    # fraction of the 0.6 km sediment, so the half-space form applies).
    scholte = scholte_speed(model.water, *sediment)
    assert abs(wet[0] / scholte - 1) < 0.01, (depth_km, wet[0], scholte)
    # Water only slows the mode; by less than 1.5 % at 60 s for these depths.
    assert np.all(wet <= dry * (1 + 1e-6)) and wet[-1] > 0.985 * dry[-1]
    # No mode jump: the same values from other search settings (differences up to 3e-6
    # come from the root refinement; 1e-5 is a margin).
    single = np.array([phase_velocity(model, [t])[0] for t in periods[::3]])
    fine = phase_velocity(model, periods, dc=0.0005)
    late = phase_velocity(model, periods[30:])
    for other, ours in ((single, wet[::3]), (fine, wet), (late, wet[30:])):
        assert np.allclose(other, ours, rtol=1e-5), depth_km
    # Largest slowdown (quoted on the page): ~42 % near 4.4 s for every depth (the
    # sediment-controlled part), 63 % near 8.3 s under 4 km of water.
    ratio = wet / dry
    assert abs(ratio[np.argmin(np.abs(periods - 4.44))] - 0.58) < 0.01
    if depth_km == 4.0:
        assert ratio.min() < 0.38 and 8.0 < periods[np.argmin(ratio)] < 8.5
    print(
        f"water {depth_km:g} km: smallest ratio {ratio.min():.3f} "
        f"at {periods[np.argmin(ratio)]:.2f} s"
    )
    ax_a.plot(periods, wet, color=color, lw=1.2, label=f"water {depth_km:g} km")
    ax_b.plot(periods, wet / dry, color=color, lw=1.2)
    if depth_km == 2.0:
        # Mode 1 depends on the search step below ~2.8 s: the default dc steps over
        # roots there (the docstring's "a root closer than dc to another can be missed").
        coarse = phase_velocity(model, periods, mode=1)
        finer = phase_velocity(model, periods, mode=1, dc=0.0005)
        short, longer = periods < 2.7, (periods > 2.9) & np.isfinite(finer)
        assert np.max(np.abs(coarse[short] / finer[short] - 1)) > 0.5
        assert np.allclose(coarse[longer], finer[longer], rtol=1e-5)
        assert np.all(finer[np.isfinite(finer)] > wet[np.isfinite(finer)])
ax_a.axhline(scholte, color=_style.APPROX, ls=":", lw=1.0, label="Scholte, water/sediment")
ax_a.set_xscale("log")
ax_a.set_ylim(0, 4.8)
ax_a.set_xlabel("period (s)")
ax_a.set_ylabel("phase velocity (km/s)")
ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
_style.panel_label(ax_a, "(a)")
ax_b.set_xscale("log")
ax_b.set_xlabel("period (s)")
ax_b.set_ylabel(r"$c_\mathrm{water} / c_\mathrm{no\ water}$")
_style.panel_label(ax_b, "(b)")
fig.tight_layout()
_style.save(fig, __file__)
print(f"Scholte speed {scholte:.4f} km/s; short-period ratio to dry {wet[0] / dry[0]:.3f}")
