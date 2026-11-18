"""Constant-Q formulas on docs/attenuation.md, checked against the package with numpy.

Writes LaTeX snippets docs/generated/attenuation_*.md, which the page includes.
Each formula is written once here and asserted against
``complybay.forward.attenuation`` (tolerance and reason next to each):

- the complex velocity and gamma(Q), against ``complex_velocity``;
- Q = Re M / (-Im M) = cot(pi gamma), exactly;
- the phase velocity omega / Re k = v_ref (f / f_ref)^gamma, against
  ``complex_velocity`` and ``phase_velocity_at``;
- the decay rate Im k against omega / (2 Q c);
- a table of the first-order form 1 + ln(f / f_ref) / (pi Q) against the
  exact power law.

Run: python docs/scripts/attenuation_derivation.py
"""

from __future__ import annotations

import numpy as np
from _style import GENERATED_DIR

from complybay.forward.attenuation import (
    complex_velocity,
    dispersed,
    kjartansson_gamma,
    phase_velocity_at,
)
from complybay.model import LayeredModel


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"attenuation_{name}.md").write_text(text, encoding="utf-8")


# --- 1. Complex velocity ---------------------------------------------------------------------
_write(
    "velocity",
    "$$\n\\boxed{\\;\\tilde v(f) = v_\\mathrm{ref}\\,\\cos\\frac{\\pi\\gamma}{2}"
    "\\left(\\frac{f}{f_\\mathrm{ref}}\\right)^{\\gamma} e^{-i\\pi\\gamma/2},"
    "\\qquad \\gamma = \\frac{1}{\\pi}\\arctan\\frac{1}{Q}\\;}\n$$\n",
)
_write(
    "identities",
    "$$\nc(f) = \\frac{\\omega}{\\operatorname{Re} k}"
    " = v_\\mathrm{ref}\\left(\\frac{f}{f_\\mathrm{ref}}\\right)^{\\gamma},"
    "\\qquad \\frac{\\operatorname{Re} M}{-\\operatorname{Im} M} = \\cot(\\pi\\gamma) = Q,"
    "\\qquad \\operatorname{Im} k = \\frac{\\omega}{2 Q c}\\left[1 + O(Q^{-2})\\right]\n$$\n\n"
    "with $k = \\omega/\\tilde v$ and $M = \\rho\\tilde v^2$.\n",
)

freqs = np.geomspace(1e-3, 10.0, 81)  # Hz
worst = {"velocity": 0.0, "q": 0.0, "phase": 0.0, "decay": 0.0}
for q in (5.0, 10.0, 20.0, 50.0, 300.0):
    for v_ref, f_ref in ((0.4, 1.0), (3.5, 0.01)):
        gamma = np.arctan(1 / q) / np.pi
        v = (
            v_ref
            * np.cos(np.pi * gamma / 2)
            * (freqs / f_ref) ** gamma
            * np.exp(-0.5j * np.pi * gamma)
        )
        package = complex_velocity(v_ref, q, freqs, f_ref)
        # Same expression evaluated twice: round-off.
        worst["velocity"] = max(worst["velocity"], np.max(np.abs(package / v - 1)))
        assert np.isclose(kjartansson_gamma(q), gamma, rtol=1e-14)
        modulus = package**2  # M / rho
        worst["q"] = max(worst["q"], np.max(np.abs(modulus.real / -modulus.imag / q - 1)))
        omega = 2 * np.pi * freqs
        k = omega / package
        phase = omega / k.real
        law = v_ref * (freqs / f_ref) ** gamma
        worst["phase"] = max(worst["phase"], np.max(np.abs(phase / law - 1)))
        assert np.allclose(
            [phase_velocity_at(v_ref, q, f, f_ref) for f in freqs[::10]], law[::10], rtol=1e-13
        )
        # High-Q decay: the relative error is O(1/Q^2) (1/(4 Q^2) to leading order here);
        # the bound 1.5 / Q^2 leaves a margin over that.
        rel = k.imag / (omega / (2 * q * phase)) - 1
        assert np.all(np.abs(rel) < 1.5 / q**2), (q, np.abs(rel).max())
        worst["decay"] = max(worst["decay"], np.abs(rel).max() * q**2)
# Round-off for complex powers and products: 1e-12.
assert worst["velocity"] < 1e-12 and worst["q"] < 1e-12 and worst["phase"] < 1e-12, worst
# Elastic layers: gamma = 0, the velocity is real and equal to v_ref.
assert kjartansson_gamma(np.inf) == 0.0
assert np.array_equal(complex_velocity(2.0, np.inf, freqs, 1.0), np.full(freqs.size, 2.0))

# dispersed(): an elastic copy with the phase velocities at one frequency.
solid = LayeredModel([0.5], [1.8, 6.0], [0.3, 3.5], [1.8, 2.7], qp=[100.0, np.inf], qs=[20.0, 50.0])
low = dispersed(solid, 0.01)
assert not low.attenuating
assert np.allclose(low.vs, phase_velocity_at(solid.vs, solid.qs, 0.01, 1.0), rtol=1e-14)

# --- 2. First-order form against the power law ----------------------------------------------
rows = []
for q in (10.0, 20.0, 50.0, 100.0):
    cells = []
    for ratio in (0.01, 0.1, 10.0):
        exact = phase_velocity_at(1.0, q, ratio, 1.0)
        linear = 1 + np.log(ratio) / (np.pi * q)
        cells.append(f"{100 * (exact - 1):+.1f} ({100 * (linear - 1):+.1f})")
    rows.append(f"| {q:g} | " + " | ".join(cells) + " |")
_write(
    "first_order",
    "| $Q$ | $f/f_\\mathrm{ref}$ = 0.01 | 0.1 | 10 |\n|---|---|---|---|\n" + "\n".join(rows) + "\n",
)
# The first-order form is within 0.3 % of the power law for Q >= 20 over two decades.
for q in (20.0, 50.0, 100.0):
    for ratio in (0.01, 100.0):
        exact = phase_velocity_at(1.0, q, ratio, 1.0)
        assert abs(exact - (1 + np.log(ratio) / (np.pi * q))) < 3e-3

print(
    f"wrote {GENERATED_DIR}/attenuation_*.md; worst relative differences: velocity "
    f"{worst['velocity']:.1e}, Q {worst['q']:.1e}, phase velocity {worst['phase']:.1e}; "
    f"max Q^2 * (decay error) {worst['decay']:.2f}"
)
