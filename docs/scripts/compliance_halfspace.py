"""Figure compliance_halfspace.png: compliance of a uniform half-space against the closed form.

(a) The package's compliance for one half-space under 1, 3 and 5 km of water,
relative to the static limit, with the exact closed form and its first-order
term. (b) The closed form over the whole range of c / v_s, with package values
for half-spaces soft enough to reach it. The closed form is the one derived in
compliance_derivation.py (asserted equal to the package there).
"""

import _style
import numpy as np
from _style import plt

from complybay.forward.compliance import (
    compliance,
    halfspace_compliance,
    infragravity_cutoff,
    water_wavenumber,
)
from complybay.model import LayeredModel, OBSModel, WaterLayer

VP, VS, RHO = 4.0, 2.0, 2.5  # km/s, km/s, g/cm^3; assumed
DEPTHS_KM = (1.0, 3.0, 5.0)


def closed_form_ratio(c_over_vs, vp_over_vs):
    """eta / eta_0 of a uniform half-space (compliance_derivation.py, section 2)."""
    s = c_over_vs**2
    q_a = np.sqrt(1 - s / vp_over_vs**2)
    q_b = np.sqrt(1 - s)
    eta_mu = s * q_a / (4 * q_a * q_b - (2 - s) ** 2)  # eta * mu
    eta0_mu = vp_over_vs**2 / (2 * (vp_over_vs**2 - 1))
    return eta_mu / eta0_mu


def kappa(vp_over_vs):
    r2 = vp_over_vs**2
    return (3 * r2**2 - 4 * r2 + 3) / (4 * r2 * (r2 - 1))


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))

eta0 = halfspace_compliance(VP, VS, RHO)
for depth_km, color in zip(DEPTHS_KM, _style.SERIES, strict=False):
    model = OBSModel(LayeredModel([], [VP], [VS], [RHO]), WaterLayer(depth_km))
    f_c = infragravity_cutoff(depth_km * 1e3)
    freqs = np.geomspace(1e-4, f_c, 60)
    omega = 2 * np.pi * freqs
    c_over_vs = omega / water_wavenumber(omega, depth_km * 1e3) / (VS * 1e3)
    ax_a.plot(freqs, 100 * (closed_form_ratio(c_over_vs, VP / VS) - 1), color=color, lw=1.2)
    ax_a.plot(freqs, 100 * kappa(VP / VS) * c_over_vs**2, color=_style.APPROX, ls="--", lw=0.9)
    package = compliance(model, freqs[::4]) / eta0
    ax_a.plot(freqs[::4], 100 * (package - 1), "o", ms=3, color=color, label=f"H = {depth_km:g} km")
    ax_a.axvline(f_c, color=color, lw=0.6, ls=":")
ax_a.plot([], [], color="0.2", lw=1.2, label="closed form")
ax_a.plot([], [], color=_style.APPROX, ls="--", label=r"$\kappa\,c^2/v_s^2$")
ax_a.set_xscale("log")
ax_a.set_xlabel("frequency (Hz)")
ax_a.set_ylabel(r"$\eta/\eta_0 - 1$ (%)")
ax_a.set_title(rf"$v_p$ = {VP:g}, $v_s$ = {VS:g} km/s; dotted: $f_c$")
ax_a.legend(loc="lower left", ncols=2)
ax_a.set_ylim(-0.32, None)  # room for the legend below the curves
_style.panel_label(ax_a, "(a)")

# (b) Soft half-spaces under 4 km of water at 0.1 mHz, v_p = 2 v_s.
ratio_grid = np.linspace(1e-3, 0.999, 2000)
denominator = (
    4 * np.sqrt(1 - ratio_grid**2 / 4) * np.sqrt(1 - ratio_grid**2) - (2 - ratio_grid**2) ** 2
)
c_rayleigh = ratio_grid[np.flatnonzero(np.diff(np.sign(denominator)))[0]]
below = ratio_grid < c_rayleigh
ax_b.plot(
    ratio_grid[below],
    closed_form_ratio(ratio_grid[below], 2.0),
    color=_style.TRUTH,
    lw=1.2,
    label="closed form",
)
ax_b.plot(
    ratio_grid, 1 + kappa(2.0) * ratio_grid**2, color=_style.APPROX, ls="--", label="first order"
)
depth_m, freq = 4000.0, 1e-4
c_m_s = 2 * np.pi * freq / water_wavenumber(2 * np.pi * freq, depth_m)[0]
targets = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85])
package = []
for target in targets:
    vs = c_m_s / target / 1e3
    model = OBSModel(LayeredModel([], [2 * vs], [vs], [RHO]), WaterLayer(depth_m / 1e3))
    package.append(compliance(model, [freq])[0] / halfspace_compliance(2 * vs, vs, RHO))
ax_b.plot(targets, package, "o", ms=4, color=_style.PACKAGE, label="package")


def _package_ratio(c_over_vs: float) -> float:
    vs = c_m_s / c_over_vs / 1e3
    model = OBSModel(LayeredModel([], [2 * vs], [vs], [RHO]), WaterLayer(depth_m / 1e3))
    with np.errstate(invalid="ignore"):
        return compliance(model, [freq])[0] / halfspace_compliance(2 * vs, vs, RHO)


# Failure modes shown shaded: between c_R and v_s the result changes sign (it passes
# through the Rayleigh resonance); at c > v_s the half-space has no decaying S wave and
# the propagator returns NaN.
assert _package_ratio(0.5 * (c_rayleigh + 1.0)) < 0
assert np.isnan(_package_ratio(1.05))
ax_b.axvspan(c_rayleigh, 1.0, color=_style.HIGHLIGHT, alpha=0.12, lw=0)
ax_b.text(
    c_rayleigh - 0.02, 4.5, r"$c > c_R$: invalid", color=_style.HIGHLIGHT, ha="right", fontsize=8
)
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0.9, 5)
ax_b.set_xlabel(r"$c / v_s$")
ax_b.set_ylabel(r"$\eta/\eta_0$")
ax_b.set_title(r"$v_p = 2 v_s$")
ax_b.legend(loc="center left")
_style.panel_label(ax_b, "(b)")

fig.tight_layout()
_style.save(fig, __file__)
print(f"Rayleigh speed for v_p = 2 v_s: {c_rayleigh:.4f} v_s")
