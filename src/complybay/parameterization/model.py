"""Continuous seafloor models: segments separated by (possibly gradational) interfaces.

Adapted from SeisForge (``seisforge.inv.model``, commit 019d568). A
:class:`ParameterizedModel` is an ordered stack of :class:`Segment` objects
below the seafloor, separated by :class:`Interface` objects with absolute
depths (km below the seafloor). Each segment has

- a Vs profile on its own local coordinate, so it stretches when an interface
  moves;
- optionally its own Vp profile or a Vp/Vs profile; otherwise Vp follows the
  segment's Vp relation (a sediment needs its own Vp: its Vp stays near
  1.5-2 km/s while Vs grows several-fold, so neither a relation nor a constant
  Vp/Vs describes it);
- a density relation applied to the segment's Vp;
- optionally Qp and Qs profiles (constant quality factors of
  :mod:`complybay.forward.attenuation`; absent = elastic). With Q the model's
  velocities are phase velocities at ``q_reference_frequency`` (Hz).

Inside a gradational interface every property is blended with the same weight,

    v(z) = (1 - S(z)) v_upper(z) + S(z) v_lower(z),

and attenuation as 1/Q (an elastic neighbour contributes 1/Q = 0),

where each neighbour is evaluated with depths clamped to its own segment.
Conversion to finite layers lives in :mod:`complybay.parameterization.layering`.
Differences from SeisForge: the optional sampled Vp or Vp/Vs profile, density
always computed from Vp, and an interface layout only (no fixed-boundary
segment layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from complybay.model.layered import Q_REFERENCE_FREQUENCY
from complybay.parameterization.bspline import evaluate_bspline
from complybay.parameterization.scaling import ScalingConfig
from complybay.parameterization.transitions import (
    TRANSITION_SHAPES,
    transition_half_support_km,
    transition_weight,
)


class Profile(Protocol):
    """A property (Vs, Vp or Vp/Vs) inside one segment, on local depth 0..thickness."""

    def evaluate(self, z_local: np.ndarray, *, thickness: float) -> np.ndarray: ...


@dataclass(frozen=True)
class ConstantProfile:
    value: float

    def evaluate(self, z_local, *, thickness: float) -> np.ndarray:
        return np.full_like(np.asarray(z_local, dtype=float), self.value)


@dataclass(frozen=True)
class GradientProfile:
    """Linear from ``top`` at the segment top to ``bottom`` at its bottom."""

    top: float
    bottom: float

    def evaluate(self, z_local, *, thickness: float) -> np.ndarray:
        fraction = np.asarray(z_local, dtype=float) / thickness
        return self.top + (self.bottom - self.top) * fraction


@dataclass(frozen=True)
class BSplineProfile:
    """B-spline on the segment's normalized coordinate (open-clamped knots).

    The first and last coefficients are the values at the segment top and
    bottom; interior coefficients are basis weights, not point values.
    """

    coefficients: tuple[float, ...]
    degree: int = 3
    knot_spacing: str = "geometric"
    knot_alpha: float = 2.0

    def __post_init__(self):
        coefficients = tuple(float(c) for c in self.coefficients)
        if not coefficients or not np.all(np.isfinite(coefficients)):
            raise ValueError("B-spline coefficients must be finite and non-empty.")
        if min(coefficients) <= 0:
            raise ValueError("B-spline coefficients must be positive.")
        if self.degree < 0 or self.knot_alpha <= 0:
            raise ValueError("B-spline degree must be >= 0 and knot_alpha > 0.")
        object.__setattr__(self, "coefficients", coefficients)

    def evaluate(self, z_local, *, thickness: float) -> np.ndarray:
        return evaluate_bspline(
            z_local,
            np.asarray(self.coefficients),
            degree=self.degree,
            domain=(0.0, float(thickness)),
            spacing=self.knot_spacing,
            alpha=self.knot_alpha,
        )


@dataclass(frozen=True)
class Segment:
    """Depth interval [top, bottom] (km) with its own profiles and relations."""

    top: float
    bottom: float
    vs: Profile
    vp_vs: Profile | None = None  # Vp = vs * vp_vs
    name: str | None = None
    scaling: ScalingConfig | None = None  # None: the model-level relations
    vp: Profile | None = None  # Vp directly; with neither vp nor vp_vs, the Vp relation
    qp: Profile | None = None  # None: elastic
    qs: Profile | None = None

    def __post_init__(self):
        if self.top < 0 or self.bottom <= self.top:
            raise ValueError("Segment needs 0 <= top < bottom.")
        if self.vp is not None and self.vp_vs is not None:
            raise ValueError("A segment takes a vp or a vp_vs profile, not both.")

    @property
    def thickness(self) -> float:
        return self.bottom - self.top

    def properties(self, z, scaling: ScalingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Vs, Vp, rho) at depths ``z`` clamped to the segment."""
        z_local = np.clip(np.asarray(z, dtype=float), self.top, self.bottom) - self.top
        vs = self.vs.evaluate(z_local, thickness=self.thickness)
        if self.vp is not None:
            vp = self.vp.evaluate(z_local, thickness=self.thickness)
        elif self.vp_vs is not None:
            vp = vs * self.vp_vs.evaluate(z_local, thickness=self.thickness)
        else:
            vp = scaling.vp_from_vs(vs)
        return vs, vp, scaling.rho_from_vp(vp)

    def inverse_q(self, z) -> tuple[np.ndarray, np.ndarray]:
        """(1/Qp, 1/Qs) at depths ``z`` clamped to the segment; 0 where elastic."""
        z_local = np.clip(np.asarray(z, dtype=float), self.top, self.bottom) - self.top
        out = []
        for profile in (self.qp, self.qs):
            if profile is None:
                out.append(np.zeros(z_local.shape))
                continue
            q = profile.evaluate(z_local, thickness=self.thickness)
            if np.any(q <= 0):
                raise ValueError("Q must be positive.")
            out.append(1.0 / q)
        return out[0], out[1]


@dataclass(frozen=True)
class Interface:
    """Boundary at ``depth_km`` with 10-90 % transition width ``width_km`` (0 = hard)."""

    depth_km: float
    width_km: float = 0.0
    shape: str = "sigmoid"
    name: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.depth_km) or self.depth_km <= 0:
            raise ValueError("Interface depth must be finite and positive.")
        if not np.isfinite(self.width_km) or self.width_km < 0:
            raise ValueError("Interface width must be finite and non-negative.")
        if self.shape not in TRANSITION_SHAPES:
            raise ValueError(f"Interface shape must be one of {TRANSITION_SHAPES}.")

    @property
    def is_hard(self) -> bool:
        return self.width_km == 0

    @property
    def half_support_km(self) -> float:
        return transition_half_support_km(self.width_km, self.shape)

    @property
    def support_km(self) -> tuple[float, float]:
        half = self.half_support_km
        return self.depth_km - half, self.depth_km + half

    def weight(self, z) -> np.ndarray:
        """Weight of the segment below: 0 above, 1 below the transition."""
        return transition_weight(z, self.depth_km, self.width_km, self.shape)


@dataclass(frozen=True)
class ParameterizedModel:
    """Segments from the seafloor (0 km) to ``zmax``, separated by interfaces.

    ``interfaces[i]`` separates ``segments[i]`` and ``segments[i + 1]``; its
    depth equals their shared boundary. ``interfaces=None`` makes every
    boundary a hard step. Transition supports may not reach the seafloor,
    overlap each other, or extend below ``zmax``.
    """

    segments: tuple[Segment, ...]
    interfaces: tuple[Interface, ...] | None = None
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    q_reference_frequency: float = Q_REFERENCE_FREQUENCY  # Hz, with attenuation only

    def __post_init__(self):
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("A model needs at least one segment.")
        if not np.isclose(segments[0].top, 0.0):
            raise ValueError("The first segment must start at the seafloor (0 km).")
        for upper, lower in zip(segments[:-1], segments[1:], strict=True):
            if not np.isclose(upper.bottom, lower.top):
                raise ValueError("Segments must be contiguous and ordered.")
        if self.interfaces is None:
            interfaces = tuple(Interface(depth_km=s.top) for s in segments[1:])
        else:
            interfaces = tuple(self.interfaces)
            if len(interfaces) != len(segments) - 1:
                raise ValueError("A model needs one interface between each pair of segments.")
            for interface, segment in zip(interfaces, segments[1:], strict=True):
                if not np.isclose(interface.depth_km, segment.top):
                    raise ValueError("Interface depths must equal the segment boundaries.")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "interfaces", interfaces)
        _check_transition_supports(interfaces, segments[-1].bottom)

    @property
    def zmax(self) -> float:
        return self.segments[-1].bottom

    def interface_depths(self) -> np.ndarray:
        return np.array([i.depth_km for i in self.interfaces], dtype=float)

    def discontinuities(self) -> np.ndarray:
        """Depths of hard (zero-width) interfaces."""
        return np.array([i.depth_km for i in self.interfaces if i.is_hard], dtype=float)

    def segment_scaling(self, index: int) -> ScalingConfig:
        return self.segments[index].scaling or self.scaling

    def segment_index(self, z) -> np.ndarray:
        """Segment owning each depth when every interface is treated as hard."""
        z = np.asarray(z, dtype=float)
        if np.any(z < 0) or np.any(z > self.zmax):
            raise ValueError("Requested depths fall outside the model range.")
        return np.searchsorted(self.interface_depths(), z, side="right")

    def evaluate(self, z) -> np.ndarray:
        """Vs (km/s) at depths ``z``, gradational interfaces included."""
        return self.evaluate_properties(z)[0]

    def evaluate_properties(self, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Vs, Vp, rho) at depths ``z``; each blended across transitions."""
        z = np.asarray(z, dtype=float)
        owner = self.segment_index(z)
        out = tuple(np.empty(z.shape) for _ in range(3))
        for index in range(len(self.segments)):
            mask = owner == index
            if np.any(mask):
                for target, part in zip(out, self._properties(index, z[mask]), strict=True):
                    target[mask] = part
        for index, interface in enumerate(self.interfaces):
            if interface.is_hard:
                continue
            top, bottom = interface.support_km
            mask = (z > top) & (z < bottom)
            if not np.any(mask):
                continue
            weight = interface.weight(z[mask])
            upper = self._properties(index, z[mask])
            lower = self._properties(index + 1, z[mask])
            for target, above, below in zip(out, upper, lower, strict=True):
                target[mask] = (1.0 - weight) * above + weight * below
        return out

    @property
    def attenuating(self) -> bool:
        return any(s.qp is not None or s.qs is not None for s in self.segments)

    def evaluate_inverse_q(self, z) -> tuple[np.ndarray, np.ndarray]:
        """(1/Qp, 1/Qs) at depths ``z``, blended across transitions like the velocities."""
        z = np.asarray(z, dtype=float)
        owner = self.segment_index(z)
        out = (np.empty(z.shape), np.empty(z.shape))
        for index, segment in enumerate(self.segments):
            mask = owner == index
            if np.any(mask):
                for target, part in zip(out, segment.inverse_q(z[mask]), strict=True):
                    target[mask] = part
        for index, interface in enumerate(self.interfaces):
            if interface.is_hard:
                continue
            top, bottom = interface.support_km
            mask = (z > top) & (z < bottom)
            if not np.any(mask):
                continue
            weight = interface.weight(z[mask])
            upper = self.segments[index].inverse_q(z[mask])
            lower = self.segments[index + 1].inverse_q(z[mask])
            for target, above, below in zip(out, upper, lower, strict=True):
                target[mask] = (1.0 - weight) * above + weight * below
        return out

    def _properties(self, index: int, z: np.ndarray):
        try:
            return self.segments[index].properties(z, self.segment_scaling(index))
        except ValueError as exc:
            label = self.segments[index].name or f"segment {index}"
            raise ValueError(f"{label}: {exc}") from exc


def _check_transition_supports(interfaces: tuple[Interface, ...], zmax: float) -> None:
    previous_bottom = 0.0
    for interface in interfaces:
        top, bottom = interface.support_km
        label = interface.name or f"interface at {interface.depth_km:g} km"
        if top < previous_bottom and not np.isclose(top, previous_bottom):
            raise ValueError(
                f"{label}: transition support overlaps the seafloor or the previous interface."
            )
        previous_bottom = max(previous_bottom, bottom)
    if previous_bottom > zmax and not np.isclose(previous_bottom, zmax):
        raise ValueError("Interface transition support extends below the model bottom.")
