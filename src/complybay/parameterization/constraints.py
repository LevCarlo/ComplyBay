"""Hard physical constraints: models that fail any of them are outside the prior.

Adapted from SeisForge (``seisforge.inv.constraints``, commit 019d568). Checks
run on the continuous model, sampled every ``dz`` km, before any forward
calculation. Instead of fixed depth ranges, a constraint can be limited to
named segments: it then checks those segments' own profiles over their own
intervals, which follow interfaces whose depths are sampled (SeisForge's
depth ranges do not). Without ``segments`` it checks the whole blended model.

Config (``Constraints`` list)::

    - {type: vp_vs_range, bounds: [1.6, 12.0], segments: [sediment]}
    - {type: vs_range, bounds: [0.05, 1.5], segments: [sediment]}
    - {type: monotonic_vs, segments: [crust], tolerance: 0.0}
    - {type: boundary_non_decreasing_vs, tolerance: 0.0}
    - {type: max_vs_gradient, max_abs: 5.0, segments: [sediment]}   # km/s per km
    - {type: vp_above_water, margin: 0.0}    # sediment Vp >= water Vp (+ margin)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from complybay.model import OBSModel
from complybay.parameterization.model import ParameterizedModel


class Constraint(Protocol):
    name: str

    def check(self, model: ParameterizedModel, obs: OBSModel) -> bool: ...


def _profiles(model: ParameterizedModel, segments: tuple[str, ...] | None, dz: float):
    """[(z, vs, vp)] sampled every ``dz`` km.

    Without ``segments``: one entry for the whole model, transitions blended.
    With ``segments``: one entry per named segment, from that segment's own
    profiles over its own interval, so the check describes that material and
    not its blend with a neighbour inside a transition.
    """
    if dz <= 0:
        raise ValueError("Constraint sampling dz must be positive.")

    def grid(top, bottom):
        return np.linspace(top, bottom, max(1, int(np.ceil((bottom - top) / dz))) + 1)

    if segments is None:
        # Both sides of every boundary: its depth belongs to the segment below.
        edges = [s.top for s in model.segments] + [model.zmax]
        above = [s.bottom - 1e-9 * s.thickness for s in model.segments[:-1]]
        z = np.unique(np.concatenate([grid(0.0, model.zmax), edges, above]))
        vs, vp, _ = model.evaluate_properties(z)
        return [(z, vs, vp)]
    out = []
    for index, segment in enumerate(model.segments):
        if segment.name in segments:
            z = grid(segment.top, segment.bottom)
            vs, vp, _ = segment.properties(z, model.segment_scaling(index))
            out.append((z, vs, vp))
    return out


def _segments(value) -> tuple[str, ...] | None:
    return None if value is None else tuple(str(v) for v in value)


@dataclass(frozen=True)
class VpVsRange:
    bounds: tuple[float, float]
    segments: tuple[str, ...] | None = None
    dz: float = 0.05
    name: str = "vp_vs_range"

    def check(self, model, obs) -> bool:
        lo, hi = self.bounds
        return all(
            np.all((vp / vs >= lo) & (vp / vs <= hi))
            for _, vs, vp in _profiles(model, self.segments, self.dz)
        )


@dataclass(frozen=True)
class VsRange:
    bounds: tuple[float, float]
    segments: tuple[str, ...] | None = None
    dz: float = 0.05
    name: str = "vs_range"

    def check(self, model, obs) -> bool:
        lo, hi = self.bounds
        return all(
            np.all((vs >= lo) & (vs <= hi)) for _, vs, _ in _profiles(model, self.segments, self.dz)
        )


@dataclass(frozen=True)
class MonotonicVs:
    """Vs must not decrease with depth by more than ``tolerance`` between samples."""

    segments: tuple[str, ...] | None = None
    tolerance: float = 0.0
    dz: float = 0.05
    name: str = "monotonic_vs"

    def check(self, model, obs) -> bool:
        return all(
            np.all(np.diff(vs) >= -self.tolerance)
            for _, vs, _ in _profiles(model, self.segments, self.dz)
        )


@dataclass(frozen=True)
class BoundaryNonDecreasingVs:
    """Vs must not drop across any interface (compares the two segment profiles)."""

    tolerance: float = 0.0
    name: str = "boundary_non_decreasing_vs"

    def check(self, model, obs) -> bool:
        for index, interface in enumerate(model.interfaces):
            z = np.array([interface.depth_km])
            above = model.segments[index].properties(z, model.segment_scaling(index))[0][0]
            below = model.segments[index + 1].properties(z, model.segment_scaling(index + 1))[0][0]
            if above - below > self.tolerance:
                return False
        return True


@dataclass(frozen=True)
class MaxVsGradient:
    """|dVs/dz| <= ``max_abs`` (km/s per km).

    Without ``segments`` this includes transition gradients (about 1.1 dVs / width
    for a sigmoid), so it also bounds jump / width; with ``segments`` it only
    sees the segments' own profiles.
    """

    max_abs: float
    segments: tuple[str, ...] | None = None
    dz: float = 0.05
    name: str = "max_vs_gradient"

    def check(self, model, obs) -> bool:
        return all(
            np.all(np.abs(np.diff(vs) / np.diff(z)) <= self.max_abs)
            for z, vs, _ in _profiles(model, self.segments, self.dz)
        )


@dataclass(frozen=True)
class VpAboveWater:
    """Vp of the layers below the seafloor >= water Vp + ``margin`` (km/s).

    Marine sediment Vp is close to, but not below, the water velocity. Passes
    for a land station (no water).
    """

    margin: float = 0.0
    segments: tuple[str, ...] | None = None
    dz: float = 0.05
    name: str = "vp_above_water"

    def check(self, model, obs) -> bool:
        if obs.water is None:
            return True
        floor = obs.water.vp + self.margin
        return all(np.all(vp >= floor) for _, _, vp in _profiles(model, self.segments, self.dz))


@dataclass(frozen=True)
class ConstraintEvaluation:
    valid: bool
    failed: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintSuite:
    constraints: tuple[Constraint, ...] = ()

    def evaluate(self, model: ParameterizedModel, obs: OBSModel) -> ConstraintEvaluation:
        failed = []
        for constraint in self.constraints:
            try:
                valid = constraint.check(model, obs)
            except ValueError:
                valid = False
            if not valid:
                failed.append(constraint.name)
        return ConstraintEvaluation(valid=not failed, failed=tuple(failed))


def _pair(value) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("Expected [lower, upper].")
    return float(value[0]), float(value[1])


def constraints_from_config(items: Sequence[dict] | None) -> ConstraintSuite:
    """Build a suite from a list of ``{type: ..., ...}`` mappings."""
    built = []
    for item in items or ():
        kind = str(item["type"]).lower()
        segments = _segments(item.get("segments"))
        dz = float(item.get("dz", 0.05))
        if kind == "vp_vs_range":
            built.append(VpVsRange(_pair(item["bounds"]), segments, dz))
        elif kind == "vs_range":
            built.append(VsRange(_pair(item["bounds"]), segments, dz))
        elif kind == "monotonic_vs":
            built.append(MonotonicVs(segments, float(item.get("tolerance", 0.0)), dz))
        elif kind == "boundary_non_decreasing_vs":
            built.append(BoundaryNonDecreasingVs(float(item.get("tolerance", 0.0))))
        elif kind == "max_vs_gradient":
            built.append(MaxVsGradient(float(item["max_abs"]), segments, dz))
        elif kind == "vp_above_water":
            built.append(VpAboveWater(float(item.get("margin", 0.0)), segments, dz))
        else:
            raise ValueError(f"Unknown constraint type: {item['type']!r}")
    return ConstraintSuite(tuple(built))
