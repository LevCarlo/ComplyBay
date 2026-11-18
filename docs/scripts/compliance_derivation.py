"""Closed forms on docs/compliance.md, checked against the package with numpy.

Writes LaTeX snippets docs/generated/compliance_*.md, which the page includes.
Each formula is written once here, evaluated with numpy, and asserted against
an independent computation (tolerance and reason next to each):

- gravity-wave dispersion relation and infragravity cutoff, against
  ``water_wavenumber`` and ``infragravity_cutoff``;
- compliance of a uniform half-space under a pressure wave of phase speed c
  (derived here, appendix of the page), against ``compliance`` (the Fortran
  propagator); its static limit against ``halfspace_compliance``; its
  first-order coefficient kappa against the small-c limit of the closed form;
- |eta| changes at second order in 1/Q, on the static half-space;
- the error of the ``wavenumber="poly"`` option (a table of measured numbers).

Run: python docs/scripts/compliance_derivation.py
"""

from __future__ import annotations

import numpy as np
from _models import sediment_over_crust
from _style import GENERATED_DIR

from complybay.forward.compliance import (
    G_DEFAULT,
    compliance,
    halfspace_compliance,
    infragravity_cutoff,
    water_wavenumber,
)
from complybay.model import LayeredModel, OBSModel, WaterLayer


def _write(name: str, latex: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"compliance_{name}.md").write_text(latex, encoding="utf-8")


# --- 1. Gravity waves: dispersion relation and cutoff -------------------------------------
_write(
    "dispersion",
    "$$\n\\boxed{\\;\\omega^2 = g k \\tanh(kH)\\;}\n$$\n\n"
    "$$\n\\frac{p(z=-H)}{p(z=0)} = \\frac{1}{\\cosh(kH)},"
    "\\qquad c = \\frac{\\omega}{k} \\le \\sqrt{g H}\n$$\n",
)
_write(
    "cutoff",
    "$$\n\\boxed{\\;f_c = \\sqrt{\\frac{g \\tanh(2\\pi)}{2\\pi H}}"
    " \\approx \\sqrt{\\frac{g}{2\\pi H}}\\;}\n$$\n\n"
    "$$\n\\frac{p(z=-H)}{p(z=0)}\\Big|_{f_c} = \\frac{1}{\\cosh 2\\pi}"
    f" = \\frac{{1}}{{{np.cosh(2 * np.pi):.0f}}}\n$$\n",
)
# Exact k solves the relation and c <= sqrt(gH); f_c is the closed form and gives kH = 2 pi.
# Double-precision evaluations of the same expressions: 1e-12 (1e-10 for the iterated root).
for depth_m in (100.0, 2000.0, 6000.0):
    omega = 2 * np.pi * np.geomspace(1e-4, 0.1, 200)
    k = water_wavenumber(omega, depth_m)
    assert np.allclose(omega**2, G_DEFAULT * k * np.tanh(k * depth_m), rtol=1e-12)
    assert np.all(omega / k <= np.sqrt(G_DEFAULT * depth_m) * (1 + 1e-12))
    f_c = np.sqrt(G_DEFAULT * np.tanh(2 * np.pi) / (2 * np.pi * depth_m))
    assert np.isclose(infragravity_cutoff(depth_m), f_c, rtol=1e-12)
    assert np.isclose(
        water_wavenumber(2 * np.pi * f_c, depth_m)[0] * depth_m, 2 * np.pi, rtol=1e-10
    )
assert 1 - np.tanh(2 * np.pi) < 1e-5  # the deep-water form is within 1e-5

# --- 2. Uniform half-space ------------------------------------------------------------------
_write(
    "halfspace",
    "$$\n\\boxed{\\;\\eta(c) = \\frac{s\\, q_\\alpha}"
    "{\\mu \\left(4 q_\\alpha q_\\beta - (2 - s)^2\\right)}\\;}\n$$\n\n"
    "$$\ns = \\frac{c^2}{v_s^2},\\quad q_\\alpha = \\sqrt{1 - \\frac{c^2}{v_p^2}},"
    "\\quad q_\\beta = \\sqrt{1 - \\frac{c^2}{v_s^2}},\\quad \\mu = \\rho v_s^2\n$$\n",
)
_write(
    "static",
    "$$\n\\boxed{\\;\\eta_0 = \\lim_{c\\to 0} \\eta"
    " = \\frac{\\lambda + 2\\mu}{2\\mu(\\lambda + \\mu)}"
    " = \\frac{v_p^2}{2\\mu\\,(v_p^2 - v_s^2)}\\;}\n$$\n",
)


def eta_closed(c, vp, vs, rho):
    """Closed form above, SI units."""
    s = c**2 / vs**2
    q_a, q_b = np.sqrt(1 - c**2 / vp**2), np.sqrt(1 - s)
    return s * q_a / (rho * vs**2 * (4 * q_a * q_b - (2 - s) ** 2))


def eta_static(vp, vs, rho):
    mu = rho * vs**2
    lam = rho * vp**2 - 2 * mu
    return (lam + 2 * mu) / (2 * mu * (lam + mu))


def kappa(vp_over_vs):
    r2 = np.asarray(vp_over_vs, dtype=float) ** 2
    return (3 * r2**2 - 4 * r2 + 3) / (4 * r2 * (r2 - 1))


# Package against the closed forms. halfspace_compliance: same expression, round-off
# (1e-12). compliance(): the Fortran propagator solves the same boundary-value problem
# independently; the largest difference seen is ~3e-11, so 1e-9 is a round-off margin.
worst_static = worst_dynamic = 0.0
for vp, vs, rho, depth_km in [
    (4.0, 2.0, 2.5, 3.0),
    (1.8, 0.3, 1.8, 4.0),
    (6.5, 3.7, 2.9, 0.05),
    (2.0, 1.0, 2.0, 5.0),
]:
    si = (vp * 1e3, vs * 1e3, rho * 1e3)
    assert np.isclose(
        eta_static(*si),
        si[0] ** 2 / (2 * si[2] * si[1] ** 2 * (si[0] ** 2 - si[1] ** 2)),
        rtol=1e-12,
    )
    worst_static = max(worst_static, abs(halfspace_compliance(vp, vs, rho) / eta_static(*si) - 1))
    model = OBSModel(LayeredModel([], [vp], [vs], [rho]), WaterLayer(depth_km))
    freqs = np.geomspace(1e-4, infragravity_cutoff(depth_km * 1e3), 40)
    omega = 2 * np.pi * freqs
    c = omega / water_wavenumber(omega, depth_km * 1e3)
    worst_dynamic = max(
        worst_dynamic, np.max(np.abs(compliance(model, freqs) / eta_closed(c, *si) - 1))
    )
    # Static limit and kappa: (eta/eta0 - 1)/s -> kappa as s -> 0; the residual is O(s).
    s = np.array([1e-4, 1e-3])
    slope = (eta_closed(np.sqrt(s) * si[1], *si) / eta_static(*si) - 1) / s
    assert np.all(np.abs(slope - kappa(vp / vs)) < 5 * s), (slope, kappa(vp / vs))
assert worst_static < 1e-12, worst_static
assert worst_dynamic < 1e-9, worst_dynamic

# Ranges of kappa and of the Rayleigh speed (root of the denominator) for Poisson's
# ratio 0 to 0.5.
nu = np.linspace(0.0, 0.499, 500)
ratios = np.sqrt((2 - 2 * nu) / (1 - 2 * nu))
s_grid = np.linspace(0.5, 0.999999, 20001)
c_rayleigh = []
for r in ratios:
    d = 4 * np.sqrt(1 - s_grid / r**2) * np.sqrt(1 - s_grid) - (2 - s_grid) ** 2
    c_rayleigh.append(np.sqrt(s_grid[np.flatnonzero(np.diff(np.sign(d)))[0]]))
c_rayleigh = np.array(c_rayleigh)
_write(
    "correction",
    "$$\n\\frac{\\eta}{\\eta_0} = 1 + \\kappa\\,\\frac{c^2}{v_s^2}"
    " + O\\!\\left(\\frac{c^4}{v_s^4}\\right),\n$$\n\n"
    "$$\n\\kappa = \\frac{3 v_p^4 - 4 v_p^2 v_s^2 + 3 v_s^4}"
    "{4 v_p^2 \\left(v_p^2 - v_s^2\\right)}\n$$\n\n"
    f"For Poisson's ratio 0 to 0.5, $\\kappa$ lies between {kappa(ratios).min():.2f} and"
    f" {kappa(ratios).max():.2f}, and the Rayleigh speed $c_R$ (the root of the"
    f" denominator of $\\eta$) between {c_rayleigh.min():.3f} and {c_rayleigh.max():.3f} $v_s$.\n",
)

# --- 3. Attenuation: |eta| changes only at O(1/Q^2) -----------------------------------------
# Complex moduli M (1 - i/Qp), mu (1 - i/Qs) (Q = Re M / -Im M to first order) in the static
# form. A first-order change would make the deviation scale as 1/Q; it scales as 1/Q^2.
M, mu = 2.5e3 * 4e3**2, 2.5e3 * 2e3**2


def abs_eta(q):
    m_c, mu_c = M * (1 - 1j / q), mu * (1 - 0.6j / q)  # different Qp and Qs
    return abs(m_c / (2 * mu_c * (m_c - mu_c)))


dev = np.array([abs_eta(q) / abs_eta(np.inf) - 1 for q in (50.0, 100.0)])
assert np.isclose(dev[0] / dev[1], 4.0, rtol=0.02), dev  # second order; 2 % for O(1/Q^3)

# --- 4. The "poly" wavenumber option: measured error ------------------------------------
rows = []
for depth_km in (0.5, 2.0, 5.0):
    depth_m = depth_km * 1e3
    freqs = np.geomspace(1e-4, infragravity_cutoff(depth_m), 2000)
    omega = 2 * np.pi * freqs
    k_exact = water_wavenumber(omega, depth_m)
    dk = water_wavenumber(omega, depth_m, method="poly") / k_exact - 1
    i = np.argmax(np.abs(dk))
    model = sediment_over_crust(depth_km)
    deta = compliance(model, freqs, wavenumber="poly") / compliance(model, freqs) - 1
    j = np.argmax(np.abs(deta))
    rows.append(
        f"| {depth_km:g} | {100 * dk[i]:+.1f} % at $kH = {k_exact[i] * depth_m:.2f}$ "
        f"| {100 * deta[j]:+.1f} % at $kH = {k_exact[j] * depth_m:.2f}$ |"
    )
_write(
    "poly",
    "| water depth (km) | largest $k$ error | largest $\\eta$ error (model of Fig. 2) |\n"
    "|---|---|---|\n" + "\n".join(rows) + "\n",
)

print(
    f"wrote {GENERATED_DIR}/compliance_*.md; package vs closed form: static {worst_static:.1e}, "
    f"dynamic half-space {worst_dynamic:.1e}; |eta| deviation at Q = 50, 100: {dev[0]:.1e}, "
    f"{dev[1]:.1e}"
)
