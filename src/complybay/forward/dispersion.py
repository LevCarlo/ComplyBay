"""Surface-wave phase velocity for a seafloor model (disba).

The water column is passed to disba as a fluid top layer (vs = 0). With a few
km of water the fundamental Rayleigh mode drops steeply at short periods
(towards the water sound speed), so mode 0 below ~10 s is strongly controlled
by water depth and the shallowest sediment; that is the physical fundamental
mode, not a separate branch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from disba import DispersionError, PhaseDispersion, PhaseSensitivity

from complybay.model import OBSModel

Wave = Literal["rayleigh", "love"]
# disba ignores the thickness of the last layer (the half-space).
_HALFSPACE_THICKNESS = 1.0
_KERNEL_PARAMETERS = {"vs": "velocity_s", "vp": "velocity_p", "rho": "density"}


def disba_arrays(model: OBSModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(thickness, vp, vs, rho) in disba's order, water first when present."""
    solid = model.solid
    thickness = np.append(solid.thickness, _HALFSPACE_THICKNESS)
    vp, vs, rho = solid.vp, solid.vs, solid.rho
    if model.water is not None:
        w = model.water
        thickness = np.concatenate(([w.thickness], thickness))
        vp = np.concatenate(([w.vp], vp))
        vs = np.concatenate(([0.0], vs))
        rho = np.concatenate(([w.rho], rho))
    return thickness, vp, vs, rho


def phase_velocity(
    model: OBSModel,
    periods,
    *,
    wave: Wave = "rayleigh",
    mode: int = 0,
    algorithm: str = "dunkin",
    dc: float = 0.005,
) -> np.ndarray:
    """Phase velocity (km/s) at ``periods`` (s); NaN where the mode has no root.

    ``dc`` is disba's phase-velocity search step (km/s); a root closer than
    ``dc`` to another can be missed.
    """
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    if np.any(periods <= 0):
        raise ValueError("Periods must be positive.")
    order = np.argsort(periods)
    solver = PhaseDispersion(*disba_arrays(model), algorithm=algorithm, dc=dc)
    out = np.full(periods.size, np.nan)
    try:
        curve = solver(periods[order], mode=mode, wave=wave)
    except DispersionError:
        return out
    # disba drops periods without a root and returns the rest sorted.
    found = dict(zip(curve.period, curve.velocity, strict=True))
    out[order] = [found.get(t, np.nan) for t in periods[order]]
    return out


def phase_velocity_kernel(
    model: OBSModel,
    period: float,
    parameter: Literal["vs", "vp", "rho"] = "vs",
    *,
    wave: Wave = "rayleigh",
    mode: int = 0,
    algorithm: str = "dunkin",
    dc: float = 0.005,
) -> np.ndarray:
    """d c / d m for each solid layer and the half-space (water layer dropped).

    Values are per layer, so they scale with layer thickness; resample the
    model to an even grid before reading depth extents off the kernel.
    """
    if parameter not in _KERNEL_PARAMETERS:
        raise ValueError(f"Unknown parameter {parameter!r}.")
    sens = PhaseSensitivity(*disba_arrays(model), algorithm=algorithm, dc=dc)
    kernel = sens(float(period), mode=mode, wave=wave, parameter=_KERNEL_PARAMETERS[parameter])
    values = np.asarray(kernel.kernel, dtype=float)
    return values[1:] if model.water is not None else values
