"""Constant-Q attenuation and its velocity dispersion.

Kjartansson (1979, JGR 84(B9), 4737-4748): a modulus M(omega) =
M_0 (-i omega / omega_0)^(2 gamma), gamma = arctan(1 / Q) / pi, has a quality
factor Q that does not depend on frequency, and is causal. For time dependence
exp(-i omega t) (the convention of :mod:`complybay.forward.response`) the
complex velocity sqrt(M / rho) is

    v~(f) = v_ref cos(pi gamma / 2) (f / f_ref)^gamma exp(-i pi gamma / 2),

with v_ref the phase velocity at f_ref. Then

- the phase velocity is omega / Re k = v_ref (f / f_ref)^gamma, so a layer is
  faster at higher frequency (about ln(f / f_ref) / (pi Q) relative);
- a plane wave exp(i (k x - omega t)) decays as exp(-omega x / (2 Q c)) for
  large Q, and Q = Re M / -Im M = cot(pi gamma) exactly.

Elastic layers have Q = inf, gamma = 0 and v~ = v_ref. Solvers that need
complex velocities (the ACF response) call :func:`complex_velocity`; solvers
for which first order in 1/Q suffices (compliance, surface-wave phase
velocity) use the real velocities at each frequency, :func:`dispersed`, the
usual physical-dispersion correction (Kanamori & Anderson 1977, Rev. Geophys.
15(1), 105-112, there for Q >> 1).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from complybay.model import LayeredModel


def kjartansson_gamma(q) -> np.ndarray:
    """gamma = arctan(1 / Q) / pi (0 for Q = inf)."""
    q = np.asarray(q, dtype=float)
    if np.any(q <= 0):
        raise ValueError("Q must be positive.")
    return np.arctan(1.0 / q) / np.pi


def complex_velocity(v_ref, q, freqs, f_ref: float) -> np.ndarray:
    """Complex velocity (km/s) at ``freqs`` (Hz) for phase velocity ``v_ref`` at ``f_ref``.

    ``v_ref`` and ``q`` are scalars or arrays that broadcast against ``freqs``.
    """
    freqs = np.asarray(freqs, dtype=float)
    if np.any(freqs <= 0) or f_ref <= 0:
        raise ValueError("Frequencies must be positive.")
    gamma = kjartansson_gamma(q)
    return (
        np.asarray(v_ref, dtype=float)
        * np.cos(0.5 * np.pi * gamma)
        * (freqs / f_ref) ** gamma
        * np.exp(-0.5j * np.pi * gamma)
    )


def phase_velocity_at(v_ref, q, freq: float, f_ref: float) -> np.ndarray:
    """Real phase velocity v_ref (f / f_ref)^gamma at one frequency (Hz)."""
    if freq <= 0 or f_ref <= 0:
        raise ValueError("Frequencies must be positive.")
    return np.asarray(v_ref, dtype=float) * (freq / f_ref) ** kjartansson_gamma(q)


def dispersed(solid: LayeredModel, freq: float) -> LayeredModel:
    """Elastic copy of ``solid`` with each layer's phase velocities at ``freq`` (Hz)."""
    f_ref = solid.q_reference_frequency
    return replace(
        solid,
        vp=phase_velocity_at(solid.vp, solid.q("qp"), freq, f_ref),
        vs=phase_velocity_at(solid.vs, solid.q("qs"), freq, f_ref),
        qp=None,
        qs=None,
    )


__all__ = ["complex_velocity", "dispersed", "kjartansson_gamma", "phase_velocity_at"]
