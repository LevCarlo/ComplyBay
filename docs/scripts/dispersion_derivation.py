"""Closed forms on docs/dispersion.md, checked against the package with numpy.

Writes docs/generated/dispersion_scholte.md and dispersion_kernel_definition.md
and asserts:

- the Scholte equation (fluid half-space over a solid half-space): its root
  equals ``phase_velocity`` of mode 0 at short periods under 30 km of water,
  for a hard and a soft seafloor;
- ``phase_velocity_kernel`` is the one-sided difference
  (c(m) - c(m / 1.025)) / (m - m / 1.025) per layer (disba's ``srfker96``,
  step dp = 0.025), recomputed here with ``phase_velocity``.

Run: python docs/scripts/dispersion_derivation.py
"""

from __future__ import annotations

import numpy as np
from _style import GENERATED_DIR
from scipy.optimize import brentq

from complybay.forward.dispersion import phase_velocity, phase_velocity_kernel
from complybay.model import LayeredModel, OBSModel, WaterLayer


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"dispersion_{name}.md").write_text(text, encoding="utf-8")


def scholte_speed(water: WaterLayer, vp: float, vs: float, rho: float) -> float:
    def f(c):
        qa, qb = np.sqrt(1 - c**2 / vp**2), np.sqrt(1 - c**2 / vs**2)
        qw = np.sqrt(1 - c**2 / water.vp**2)
        return (2 - c**2 / vs**2) ** 2 - 4 * qa * qb + water.rho / rho * c**4 / vs**4 * qa / qw

    return brentq(f, 1e-3 * vs, min(vs, water.vp) * (1 - 1e-12), xtol=1e-14)


def main() -> None:
    # --- 1. Scholte wave ------------------------------------------------------------------
    _write(
        "scholte",
        "$$\n\\boxed{\\;\\left(2 - \\frac{c^2}{v_s^2}\\right)^2 - 4 q_\\alpha q_\\beta"
        " + \\frac{\\rho_w}{\\rho}\\,\\frac{c^4}{v_s^4}\\,\\frac{q_\\alpha}{q_w} = 0\\;}\n$$\n\n"
        "$$\nq_\\alpha = \\sqrt{1 - \\frac{c^2}{v_p^2}},\\quad"
        " q_\\beta = \\sqrt{1 - \\frac{c^2}{v_s^2}},\\quad"
        " q_w = \\sqrt{1 - \\frac{c^2}{v_{p,w}^2}},\\qquad c < \\min(v_s, v_{p,w})\n$$\n",
    )

    water = WaterLayer(30.0)  # deep enough that 0.5-1 s waves do not reach the sea surface
    worst = 0.0
    for vp, vs, rho in [(5.2, 3.0, 2.7), (1.8, 0.4, 1.8), (1.8, 0.3, 1.8)]:
        model = OBSModel(LayeredModel([], [vp], [vs], [rho]), water)
        expected = scholte_speed(water, vp, vs, rho)
        got = phase_velocity(model, [0.5, 1.0], dc=0.001)
        worst = max(worst, np.max(np.abs(got / expected - 1)))
    # Largest difference seen 7e-7 (disba's root refinement); 1e-5 is a margin over it.
    assert worst < 1e-5, worst
    scholte_soft = scholte_speed(WaterLayer(2.0), 1.8, 0.3, 1.8)
    _write(
        "scholte_value",
        f"For water ($v_{{p,w}}$ = 1.5 km/s, $\\rho_w$ = 1.03 g/cm³) over the sediment of the"
        f" docs model ($v_p$ = 1.8, $v_s$ = 0.3 km/s, $\\rho$ = 1.8 g/cm³), $c$ ="
        f" {scholte_soft:.4f} km/s = {scholte_soft / 0.3:.3f} $v_s$.\n",
    )

    # --- 2. What phase_velocity_kernel returns ------------------------------------------
    _write(
        "kernel_definition",
        "$$\nK_i = \\frac{c(m) - c(m_i \\to m_i/1.025)}{m_i - m_i/1.025}"
        "\\;\\approx\\; \\frac{\\partial c}{\\partial m_i}\n$$\n",
    )
    solid = LayeredModel(
        thickness=[0.5, 6.5, 53.0],
        vp=[2.0, 6.5, 8.1, 7.9],
        vs=[0.5, 3.6, 4.55, 4.3],
        rho=[1.8, 2.9, 3.35, 3.35],
    )  # assumed
    model = OBSModel(solid, WaterLayer(4.0))
    period = 20.0
    c0 = phase_velocity(model, [period])[0]
    for parameter in ("vs", "vp", "rho"):
        kernel = phase_velocity_kernel(model, period, parameter)
        for i in range(solid.n_layers + 1):
            m = solid.property(parameter)[i]
            c1 = phase_velocity(model.with_solid(solid.scaled(parameter, i, 1 / 1.025)), [period])[
                0
            ]
            one_sided = (c1 - c0) / (m / 1.025 - m)
            # The same disba root searches; 1e-6 allows for differences in root refinement.
            assert np.isclose(kernel[i], one_sided, rtol=1e-6, atol=1e-9), (parameter, i)
    print(
        f"wrote {GENERATED_DIR}/dispersion_*.md; Scholte vs package: {worst:.1e}; "
        f"Scholte speed water/sediment: {scholte_soft:.4f} km/s"
    )


if __name__ == "__main__":
    main()
