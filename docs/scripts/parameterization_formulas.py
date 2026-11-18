"""Formulas on docs/parameterization.md, checked against the package with numpy.

Writes LaTeX snippets docs/generated/parameterization_*.md, which the page
includes. Each formula is written once here, evaluated with numpy, and asserted
against the package (tolerance and reason next to each):

- the blend of two segments inside a gradational interface, for Vs, Vp, rho
  and 1/Q, against ``ParameterizedModel.evaluate_properties`` and
  ``evaluate_inverse_q``;
- the sigmoid and smootherstep weights, their 10-90 % width, support and
  steepest slope, against ``transition_weight``;
- the open-clamped B-spline profile and its knots, against ``bspline_knots``
  and ``BSplineProfile``;
- the Vp and density relations, from one coefficient list each, against
  ``complybay.parameterization.scaling``;
- the parameter priors, against their ``logpdf``;
- the vertical travel-time error of a layering, against ``layering_report``
  (integral by scipy quad here, Gauss-Legendre in the package).

Run: python docs/scripts/parameterization_formulas.py
"""

from __future__ import annotations

import numpy as np
from _style import GENERATED_DIR
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.optimize import brentq

from complybay.parameterization import (
    BSplineProfile,
    ConstantProfile,
    DiscretizationConfig,
    GradientProfile,
    HalfNormalPrior,
    Interface,
    NormalPrior,
    ParameterizedModel,
    ScalingConfig,
    Segment,
    UniformPrior,
    discretize,
    layering_report,
    transition_half_support_km,
    transition_weight,
)
from complybay.parameterization.bspline import bspline_knots
from complybay.parameterization.scaling import (
    brocher_rho_from_vp,
    brocher_vp_from_vs,
    castagna_vp_from_vs,
    gardner_rho_from_vp,
)


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"parameterization_{name}.md").write_text(text, encoding="utf-8")


# --- 1. Blend inside a gradational interface ------------------------------------------------
_write(
    "blend",
    "$$\n\\boxed{\\;m(z) = \\big(1 - S(z)\\big)\\, m_\\mathrm{u}(z) + S(z)\\, m_\\mathrm{l}(z),"
    "\\qquad m \\in \\{V_s,\\ V_p,\\ \\rho,\\ 1/Q_p,\\ 1/Q_s\\}\\;}\n$$\n",
)
sediment = Segment(
    0.0,
    1.0,
    GradientProfile(0.2, 0.9),
    vp=ConstantProfile(1.8),
    name="sediment",
    scaling=ScalingConfig(rho="gardner1974"),
    qs=ConstantProfile(20.0),
)
crust = Segment(1.0, 10.0, GradientProfile(3.0, 3.6), name="crust")  # elastic: 1/Q = 0
model = ParameterizedModel((sediment, crust), (Interface(1.0, 0.2, "sigmoid", "basement"),))
top, bottom = model.interfaces[0].support_km
z = np.linspace(top, bottom, 201)[1:-1]
s = transition_weight(z, 1.0, 0.2, "sigmoid")
upper = sediment.properties(z, model.segment_scaling(0))  # clamped to the sediment
lower = crust.properties(z, model.segment_scaling(1))  # clamped to the crust
blended = model.evaluate_properties(z)
for got, a, b in zip(blended, upper, lower, strict=True):
    assert np.allclose(got, (1 - s) * a + s * b, rtol=1e-13)  # same arithmetic
inverse_qs = model.evaluate_inverse_q(z)[1]
assert np.allclose(inverse_qs, (1 - s) / 20.0, rtol=1e-13)  # elastic neighbour: 1/Q = 0

# --- 2. Transition shapes ---------------------------------------------------------------------


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z, depth, width):
    k, h = 2 * np.log(9) / width, 2 * width
    u = np.clip(z - depth, -h, h)
    return (logistic(k * u) - logistic(-k * h)) / (logistic(k * h) - logistic(-k * h))


def smoother(x):
    return 6 * x**5 - 15 * x**4 + 10 * x**3


# 10-90 % width of the quintic on [0, 1], and the half-support it implies.
x90 = brentq(lambda x: smoother(x) - 0.9, 0.5, 1.0)
width_fraction = 2 * x90 - 1  # x90 - x10, symmetric about 0.5
smoother_half = 1 / (2 * width_fraction)


def smootherstep(z, depth, width):
    h = smoother_half * width
    return smoother(np.clip((z - depth + h) / (2 * h), 0.0, 1.0))


depth, width = 3.0, 0.4
zz = np.linspace(depth - 3 * width, depth + 3 * width, 6001)
rows = []
for shape, formula, half in (
    ("sigmoid", sigmoid, 2.0),
    ("smootherstep", smootherstep, smoother_half),
):
    package = transition_weight(zz, depth, width, shape)
    assert np.allclose(package, formula(zz, depth, width), atol=1e-14)  # same expression
    assert np.isclose(transition_half_support_km(width, shape), half * width, rtol=1e-12)
    # Exactly 0 / 1 outside the support, 0.5 at the interface, 10 / 90 % at -/+ w/2.
    assert np.all(package[zz <= depth - half * width] == 0.0)
    assert np.all(package[zz >= depth + half * width] == 1.0)
    at = transition_weight(
        np.array([depth - width / 2, depth, depth + width / 2]), depth, width, shape
    )
    assert np.allclose(at, [0.1, 0.5, 0.9], atol=1e-3)  # sigmoid: rescaling shifts them 3e-5
    slope = np.max(np.gradient(package, zz)) * width  # steepest slope in units of 1 / w
    rows.append((shape, half, slope))
k_h = 2 * np.log(9) * 2.0  # k h of the sigmoid
sigmoid_slope = (np.log(9) / 2) / (logistic(k_h) - logistic(-k_h))
smoother_slope = 15 / 16 / smoother_half
# The truncated sigmoid's slope jumps from k L(1 - L) (rescaled) to 0 at +-h.
kink = 2 * np.log(9) * logistic(-k_h) * (1 - logistic(-k_h)) / (logistic(k_h) - logistic(-k_h))
# Finite-difference maximum on a 2e-4 w grid: within 1e-4 of the analytic slope.
assert np.isclose(rows[0][2], sigmoid_slope, rtol=1e-4)
assert np.isclose(rows[1][2], smoother_slope, rtol=1e-4)
assert np.isclose(width_fraction, 0.5067270934230668, rtol=1e-12)  # the code's constant

_write(
    "transitions",
    "$$\n\\text{sigmoid:}\\quad S(z) = \\frac{L(k u) - L(-k h)}{L(k h) - L(-k h)},\\quad "
    "u = \\operatorname{clip}(z - z_i, -h, h),\\quad L(x) = \\frac{1}{1 + e^{-x}},"
    "\\quad \\boxed{\\,k = \\frac{2 \\ln 9}{w},\\ h = 2w\\,}\n$$\n\n"
    "$$\n\\text{smootherstep:}\\quad S(z) = 6x^5 - 15x^4 + 10x^3,\\quad "
    "x = \\operatorname{clip}\\Big(\\frac{z - z_i + h}{2h}, 0, 1\\Big),\\quad "
    f"\\boxed{{\\,h = \\frac{{w}}{{2 \\times {width_fraction:.5f}}} = "
    f"{smoother_half:.4f}\\,w\\,}}\n$$\n\n"
    "| shape | half-support $h$ | $S$ at $z_i \\mp w/2$ | steepest slope "
    "$\\max\\, dS/dz$ | smoothness |\n|---|---|---|---|---|\n"
    f"| `sigmoid` | $2w$ (logistic tail there: "
    f"{logistic(-k_h):.1e}) | 0.1, 0.9 (to $3\\times10^{{-5}}$) | ${sigmoid_slope:.3f}/w$ | "
    f"slope jumps by ${kink:.1e}/w$ at $\\pm h$ |\n"
    f"| `smootherstep` | ${smoother_half:.4f}\\,w$ | 0.1, 0.9 | ${smoother_slope:.3f}/w$ "
    "| $C^2$ |\n",
)

# --- 3. B-spline profile ----------------------------------------------------------------------
_write(
    "bspline",
    "$$\n\\boxed{\\;v(z) = \\sum_{j=0}^{n-1} c_j\\, B_{j,p}(x),\\qquad "
    "x = \\frac{z - z_\\mathrm{top}}{z_\\mathrm{bottom} - z_\\mathrm{top}}\\;}\n$$\n\n"
    "$$\n\\mathbf t = (\\underbrace{0,\\dots,0}_{p+1},\\ t_1, \\dots, t_m,\\ "
    "\\underbrace{1,\\dots,1}_{p+1}),\\quad m = n - p - 1,\\quad "
    "t_j = \\frac{\\sum_{i=0}^{j-1} \\alpha^i}{\\sum_{i=0}^{m} \\alpha^i}\n$$\n",
)
for n, degree, alpha in ((7, 3, 2.0), (5, 2, 1.5), (6, 3, 1.0), (2, 3, 2.0)):
    p = min(degree, n - 1)  # the degree is reduced when there are too few coefficients
    m = n - p - 1
    lengths = alpha ** np.arange(m + 1)
    interior = np.cumsum(lengths)[:-1] / lengths.sum()
    knots = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
    assert np.allclose(bspline_knots(n, degree=degree, alpha=alpha), knots, atol=1e-15)
    coefficients = np.linspace(3.0, 3.9, n) + 0.1 * np.sin(np.arange(n))
    profile = BSplineProfile(tuple(coefficients), degree=degree, knot_alpha=alpha)
    thickness = 6.0
    zl = np.linspace(0.0, thickness, 301)
    reference = BSpline(knots, coefficients, p, extrapolate=False)(zl / thickness)
    reference[-1] = coefficients[-1]  # scipy leaves x = 1 of a clamped spline undefined
    got = profile.evaluate(zl, thickness=thickness)
    assert np.allclose(got, reference, rtol=1e-12)  # same basis, independent evaluation
    assert np.isclose(got[0], coefficients[0]) and np.isclose(got[-1], coefficients[-1])

# --- 4. Vp and density relations -------------------------------------------------------------
BROCHER_VP = (0.9409, 2.0947, -0.8206, 0.2683, -0.0251)  # Vp(Vs), powers 0..4
BROCHER_RHO = (0.0, 1.6612, -0.4721, 0.0671, -0.0043, 0.000106)  # rho(Vp), powers 0..5
GARDNER = (1.741, 0.25)
CASTAGNA = (1.16, 1.36)


def poly_latex(coefficients, var: str) -> str:
    terms = []
    for power, c in enumerate(coefficients):
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        body = f"{abs(c):g}" + ("" if power == 0 else var if power == 1 else f"{var}^{power}")
        terms.append((sign, body))
    first_sign, first = terms[0]
    out = ("-" if first_sign == "-" else "") + first
    return out + "".join(f" {sign} {body}" for sign, body in terms[1:])


vs = np.linspace(0.0, 4.5, 91)
vp = np.linspace(1.5, 8.5, 141)
assert np.allclose(brocher_vp_from_vs(vs), np.polynomial.polynomial.polyval(vs, BROCHER_VP))
assert np.allclose(brocher_rho_from_vp(vp), np.polynomial.polynomial.polyval(vp, BROCHER_RHO))
assert np.allclose(gardner_rho_from_vp(vp), GARDNER[0] * vp ** GARDNER[1])
assert np.allclose(castagna_vp_from_vs(vs), CASTAGNA[0] * vs + CASTAGNA[1])
for bad in (4.6, -0.1):
    try:
        brocher_vp_from_vs(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Brocher Vp(Vs) must refuse Vs outside 0-4.5 km/s")
_write(
    "scaling",
    "| `method` | relation (km/s, g/cm³) | stated range | source |\n|---|---|---|---|\n"
    f"| Vp `brocher2005` | $V_p = {poly_latex(BROCHER_VP, 'V_s')}$ | $0 \\le V_s \\le 4.5$ "
    "(enforced) | Brocher (2005), Eq. 9 |\n"
    f"| Vp `castagna1985` | $V_p = {CASTAGNA[0]:g}\\,V_s + {CASTAGNA[1]:g}$ | none enforced "
    "| Castagna et al. (1985), mudrock line |\n"
    "| Vp `constant_vp_vs` | $V_p = r\\,V_s$, $r$ = `vp_vs` | none | - |\n"
    f"| ρ `brocher2005` | $\\rho = {poly_latex(BROCHER_RHO, 'V_p')}$ | $1.5 \\le V_p \\le 8.5$ "
    "(enforced) | Brocher (2005), Eq. 1 (fit to Nafe-Drake) |\n"
    f"| ρ `gardner1974` | $\\rho = a\\,V_p^{{\\,b}}$, $a = {GARDNER[0]:g}$, $b = {GARDNER[1]:g}$ "
    "(options `a`, `b`) | none | Gardner et al. (1974) |\n"
    "| ρ `constant` | $\\rho$ = `rho` | none | - |\n",
)

# --- 5. Parameter priors ----------------------------------------------------------------------
uniform, normal, half = (
    UniformPrior(0.2, 1.5),
    NormalPrior(0.6, 0.1, 0.2, 1.5),
    HalfNormalPrior(0.05, 0.2),
)
for theta in np.linspace(0.0, 1.6, 33):
    inside = 0.2 <= theta <= 1.5
    assert uniform.logpdf(theta) == (0.0 if inside else -np.inf)
    assert normal.logpdf(theta) == (-0.5 * ((theta - 0.6) / 0.1) ** 2 if inside else -np.inf)
    t = theta / 10  # half-normal on [0, 0.2]
    assert half.logpdf(t) == (-0.5 * (t / 0.05) ** 2 if t <= 0.2 else -np.inf)
_write(
    "priors",
    "| `type` | $\\ln p_j(\\theta_j)$ on $[a, b]$ (`bounds`) | keys |\n|---|---|---|\n"
    "| `uniform` (default) | $0$ | `bounds` |\n"
    "| `normal` | $-\\tfrac12 \\big((\\theta_j - \\mu)/\\sigma\\big)^2$ | `mean`, `sigma`, "
    "`bounds` |\n"
    "| `half_normal` | $-\\tfrac12 \\big((\\theta_j - a)/s\\big)^2$, peak at the lower bound "
    "| `scale`, `bounds` |\n",
)

# --- 6. Vertical travel-time error of a layering -----------------------------------------------
_write(
    "traveltime",
    "$$\n\\boxed{\\;\\epsilon_v = \\frac{\\sum_i h_i / v_i - \\int_0^{Z} dz / v(z)}"
    "{\\int_0^{Z} dz / v(z)}\\;},\\qquad v \\in \\{V_s, V_p\\}\n$$\n",
)
layered = discretize(model, DiscretizationConfig(layers_per_segment=4, transition_layers=4))
report = layering_report(model, layered)
zmax = float(layered.tops[-1])
breaks = np.r_[0.0, top, 1.0, bottom, zmax]
for index, label in ((0, "vs"), (1, "vp")):

    def slowness(x, index=index):
        return 1.0 / model.evaluate_properties(np.array([x]))[index][0]

    exact = sum(quad(slowness, a, b)[0] for a, b in zip(breaks[:-1], breaks[1:], strict=True))
    v_layers = layered.vs if label == "vs" else layered.vp
    eps = (np.sum(layered.thickness / v_layers[:-1]) - exact) / exact
    package = getattr(report, f"{label}_vertical_time_relative_error")
    # quad to 1e-10 against 16-point Gauss-Legendre per layer: agree to 1e-8 absolute.
    assert abs(eps - package) < 1e-8, (eps, package)

print("parameterization formulas: all checks passed")
