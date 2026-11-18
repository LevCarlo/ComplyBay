"""Figure compliance_kernel.png: Vs sensitivity kernels of compliance against depth.

(a) Sediment over crust (docs/scripts/_models.py) under 2 km of water,
resampled to 0.1 km layers: the fraction of the summed |d ln(eta) / d ln(vs)|
that lies above each depth, for four frequencies; dots mark the decay depth
1 / k of the load, dotted lines the interfaces.
(b) A uniform half-space split into thin layers: the same kernels per unit
k z, plotted against k z, fall on one curve, because in the static limit the
only length scale of the problem is 1 / k.

Checks (asserted): the density kernels of every frequency sum to -1, since
scaling all densities at fixed velocities scales all moduli and hence the
compliance by the inverse factor; the tolerance 2e-3 covers the central-
difference truncation error of order rel_step^2 = 6e-4.
"""

import _style
import numpy as np
from _models import sediment_over_crust
from _style import plt

from complybay.forward.compliance import compliance_kernel, water_wavenumber
from complybay.model import LayeredModel, OBSModel, WaterLayer

FREQS = np.array([0.004, 0.008, 0.016, 0.025])  # Hz; f_c = 0.028 Hz for 2 km water
DZ_KM = 0.1

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 4.0))

model = sediment_over_crust()
fine = model.with_solid(model.solid.resampled(DZ_KM))
k = water_wavenumber(2 * np.pi * FREQS, fine.water_depth * 1e3) * 1e3  # 1/km
kernel = compliance_kernel(fine, FREQS, "vs")
rho_sum = compliance_kernel(fine, FREQS, "rho").sum(axis=1)
assert np.all(np.abs(rho_sum + 1) < 2e-3), rho_sum

# Fraction of the total |kernel| (half-space included) that lies above each depth.
weight = np.abs(kernel)
above = np.cumsum(weight[:, :-1], axis=1) / weight.sum(axis=1, keepdims=True)
bottoms = fine.solid.tops[1:]
for j, (f, color) in enumerate(zip(FREQS, _style.SERIES, strict=False)):
    ax_a.plot(
        np.r_[0.0, above[j]], np.r_[1e-3, bottoms], color=color, lw=1.2, label=f"{1e3 * f:g} mHz"
    )
    ax_a.plot(np.interp(1 / k[j], bottoms, above[j]), 1 / k[j], "o", ms=4, color=color)
for depth in model.solid.tops[1:]:
    ax_a.axhline(depth, color="0.4", lw=0.6, ls=":")
ax_a.set_yscale("log")
ax_a.set_ylim(60, 0.05)
ax_a.set_xlim(0, 1.02)
ax_a.set_xlabel(r"fraction of $\sum |\partial \ln\eta / \partial \ln v_s|$ above depth")
ax_a.set_ylabel("depth below seafloor (km)")
ax_a.set_title("sediment over crust, H = 2 km; dots at $z = 1/k$")
ax_a.legend(loc="lower left")
_style.panel_label(ax_a, "(a)")

# (b) Uniform half-space, thin layers to 80 km (the load has decayed far above that).
VP, VS, RHO = 4.0, 2.0, 2.5  # assumed
n = 800
uniform = LayeredModel(
    np.full(n, DZ_KM), np.full(n + 1, VP), np.full(n + 1, VS), np.full(n + 1, RHO)
)
hs = OBSModel(uniform, WaterLayer(2.0))
kernel_hs = compliance_kernel(hs, FREQS, "vs")
mid_hs = uniform.tops[:-1] + DZ_KM / 2
# The curves per unit kz collapse: on kz = 0.5-5 they differ by less than 5 % of the
# peak. The spread comes from the layer thickness in kz (up to 0.25 at 25 mHz) and the
# dynamic term (c/v_s)^2 < 1 %, which both break the pure 1/k scaling.
kz = np.linspace(0.5, 5.0, 200)
curves = np.array(
    [np.interp(kz, k[j] * mid_hs, kernel_hs[j, :-1] / (k[j] * DZ_KM)) for j in range(FREQS.size)]
)
collapse = np.abs(curves - curves[0]).max() / np.abs(curves).max()
assert collapse < 0.05, collapse
# Shape, read at the lowest frequency (finest sampling in kz, 0.018 per layer): sign
# change near the seafloor and depth of the most negative value.
lowest = kernel_hs[0, :-1]
kz_lowest = k[0] * mid_hs
kz_zero = kz_lowest[np.flatnonzero(np.diff(np.sign(lowest)))[0]]
kz_min = kz_lowest[np.argmin(lowest)]
assert lowest[0] > 0 and lowest[np.argmin(lowest)] < 0
for j, (f, color) in enumerate(zip(FREQS, _style.SERIES, strict=False)):
    ax_b.plot(
        kernel_hs[j, :-1] / (k[j] * DZ_KM),
        k[j] * mid_hs,
        color=color,
        lw=1.1,
        label=f"{1e3 * f:g} mHz",
    )
ax_b.axvline(0.0, color="0.4", lw=0.6)
ax_b.set_ylim(6, 0)
ax_b.set_xlabel(r"$\partial \ln\eta / \partial \ln v_s$ per unit $kz$")
ax_b.set_ylabel(r"$k z$")
ax_b.set_title(rf"uniform half-space, $v_p$ = {VP:g}, $v_s$ = {VS:g} km/s")
ax_b.legend(loc="lower left")
_style.panel_label(ax_b, "(b)")

fig.tight_layout()
_style.save(fig, __file__)
print(
    "decay depths 1/k (km):",
    np.round(1 / k, 2),
    "; density-kernel sums:",
    np.round(rho_sum, 5),
    f"; half-space collapse: {collapse:.3f} of the peak",
    f"; half-space kernel > 0 above kz = {kz_zero:.2f}, most negative at kz = {kz_min:.2f}",
)
