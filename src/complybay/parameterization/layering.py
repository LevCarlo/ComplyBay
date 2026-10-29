"""Conversion of a :class:`ParameterizedModel` to finite layers, and its diagnostics.

Adapted from SeisForge (``seisforge.inv.layering``, commit 019d568). Forward
solvers only see the layered model, so the conversion follows fixed rules:

1. A hard interface is always a layer boundary at its exact depth.
2. The part of each segment outside transition supports is split by a target
   thickness ``dz`` or by a fixed count (``layers_per_segment``). A fixed
   count keeps the layer geometry continuous when interface depths are
   sampled; ``dz`` changes the layer count in steps, which puts steps into
   the likelihood.
3. The support of every gradational interface gets ``transition_layers``
   equal layers.
4. Each layer takes Vs, Vp and rho at its midpoint (``layer_value="midpoint"``)
   or as the depth average (``"mean"``).
5. The half-space takes the properties at the model bottom.

The result is a :class:`complybay.model.LayeredModel` (n layers + half-space).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from complybay.model import LayeredModel
from complybay.parameterization.model import ParameterizedModel


@dataclass(frozen=True)
class DiscretizationConfig:
    """How a continuous model becomes forward-model layers (depths in km)."""

    dz: float = 0.5
    layers_per_segment: int | Mapping[str, int] | None = None
    transition_layers: int = 16
    layer_value: str = "midpoint"
    zmax: float | None = None  # stop above the model bottom (half-space there)

    def __post_init__(self):
        if self.dz <= 0:
            raise ValueError("dz must be positive.")
        if self.zmax is not None and self.zmax <= 0:
            raise ValueError("zmax must be positive.")
        if self.transition_layers < 2:
            raise ValueError("transition_layers must be at least 2.")
        if self.layer_value not in ("midpoint", "mean"):
            raise ValueError("layer_value must be 'midpoint' or 'mean'.")
        counts = self.layers_per_segment
        if counts is not None:
            values = counts.values() if isinstance(counts, Mapping) else [counts]
            if any(int(v) != v or v < 1 for v in values):
                raise ValueError("layers_per_segment must be positive integers.")


@dataclass(frozen=True)
class LayerGeometry:
    """Finite-layer boundaries and the half-space top (km)."""

    tops_km: np.ndarray
    bottoms_km: np.ndarray
    zmax_km: float
    segment_index: np.ndarray = field(repr=False)  # owning segment of each layer

    @property
    def thickness_km(self) -> np.ndarray:
        return self.bottoms_km - self.tops_km


def layer_geometry(model: ParameterizedModel, config: DiscretizationConfig) -> LayerGeometry:
    """Finite-layer boundaries that respect interfaces and transition supports."""
    zmax = model.zmax if config.zmax is None else min(config.zmax, model.zmax)
    tops: list[float] = []
    bottoms: list[float] = []
    for top, bottom, n_layers in _regions(model, config):
        bottom = min(bottom, zmax)
        if bottom <= top:
            continue
        if n_layers is None:
            n_layers = max(1, int(np.ceil((bottom - top) / config.dz - 1e-9)))
        edges = np.linspace(top, bottom, n_layers + 1)
        tops.extend(edges[:-1])
        bottoms.extend(edges[1:])
        if np.isclose(bottom, zmax):
            break
    if not tops:
        raise ValueError("Discretization produced no finite layers.")
    tops_km = np.asarray(tops)
    bottoms_km = np.asarray(bottoms)
    return LayerGeometry(
        tops_km=tops_km,
        bottoms_km=bottoms_km,
        zmax_km=float(zmax),
        segment_index=model.segment_index(0.5 * (tops_km + bottoms_km)),
    )


def discretize(model: ParameterizedModel, config: DiscretizationConfig) -> LayeredModel:
    """Forward-ready layers of ``model``."""
    geometry = layer_geometry(model, config)
    if config.layer_value == "mean":
        vs, vp, rho = _layer_means(model, geometry.tops_km, geometry.bottoms_km)
    else:
        vs, vp, rho = model.evaluate_properties(0.5 * (geometry.tops_km + geometry.bottoms_km))
    vs_h, vp_h, rho_h = model.evaluate_properties(np.array([geometry.zmax_km]))
    attenuation = {}
    if model.attenuating:
        # 1/Q at layer midpoints (also with layer_value="mean"), inf where elastic.
        z = np.r_[0.5 * (geometry.tops_km + geometry.bottoms_km), geometry.zmax_km]
        with np.errstate(divide="ignore"):
            inverse_qp, inverse_qs = model.evaluate_inverse_q(z)
            attenuation = {"qp": 1.0 / inverse_qp, "qs": 1.0 / inverse_qs}
        attenuation["q_reference_frequency"] = model.q_reference_frequency
    return LayeredModel(
        thickness=geometry.thickness_km,
        vp=np.r_[vp, vp_h],
        vs=np.r_[vs, vs_h],
        rho=np.r_[rho, rho_h],
        **attenuation,
    )


@dataclass(frozen=True)
class LayeringReport:
    """How faithfully a layered model represents its continuous model.

    Vertical travel time is integral dz / v; its relative error bounds the error
    of vertical delay times such as ACF reverberations.
    """

    n_layers: int
    min_thickness_km: float
    hard_interfaces_preserved: bool
    max_interface_offset_km: float
    transition_layer_counts: tuple[int, ...]
    max_abs_vs_error_km_s: float
    vs_vertical_time_relative_error: float
    vp_vertical_time_relative_error: float

    def summary(self) -> str:
        return (
            f"{self.n_layers} layers (min {self.min_thickness_km:.4g} km); "
            f"hard interfaces preserved: {self.hard_interfaces_preserved} "
            f"(max offset {self.max_interface_offset_km:.2g} km); "
            f"transition layers: {list(self.transition_layer_counts)}; "
            f"max |dVs| {self.max_abs_vs_error_km_s:.4g} km/s; "
            f"vertical time error Vs {self.vs_vertical_time_relative_error:+.3e}, "
            f"Vp {self.vp_vertical_time_relative_error:+.3e}"
        )


def layering_report(
    model: ParameterizedModel, layered: LayeredModel, *, quadrature_points: int = 16
) -> LayeringReport:
    """Compare finite layers with the continuous model over 0..half-space top."""
    thickness = layered.thickness
    tops = layered.tops[:-1]
    bottoms = tops + thickness

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    z = 0.5 * (tops + bottoms)[:, None] + 0.5 * thickness[:, None] * nodes[None, :]
    vs_c, vp_c, _ = model.evaluate_properties(z.ravel())
    vs_c = vs_c.reshape(z.shape)
    vp_c = vp_c.reshape(z.shape)
    half = 0.5 * thickness[:, None] * weights[None, :]
    vs_time = float(np.sum(half / vs_c))
    vp_time = float(np.sum(half / vp_c))
    vs_time_layered = float(np.sum(thickness / layered.vs[:-1]))
    vp_time_layered = float(np.sum(thickness / layered.vp[:-1]))

    hard = [d for d in model.discontinuities() if d < bottoms[-1]]
    offsets = [float(np.min(np.abs(layered.tops - d))) for d in hard]
    max_offset = max(offsets, default=0.0)

    mid = 0.5 * (tops + bottoms)
    counts = []
    for interface in model.interfaces:
        if not interface.is_hard:
            top, bottom = interface.support_km
            counts.append(int(np.count_nonzero((mid > top) & (mid < bottom))))

    return LayeringReport(
        n_layers=int(layered.n_layers),
        min_thickness_km=float(np.min(thickness)),
        hard_interfaces_preserved=bool(max_offset <= 1e-9),
        max_interface_offset_km=max_offset,
        transition_layer_counts=tuple(counts),
        max_abs_vs_error_km_s=float(np.max(np.abs(vs_c - layered.vs[:-1, None]))),
        vs_vertical_time_relative_error=(vs_time_layered - vs_time) / vs_time,
        vp_vertical_time_relative_error=(vp_time_layered - vp_time) / vp_time,
    )


def _regions(model: ParameterizedModel, config: DiscretizationConfig):
    """Ordered (top, bottom, layer count or None for dz) depth regions."""
    interfaces = model.interfaces
    regions: list[tuple[float, float, int | None]] = []
    for index, segment in enumerate(model.segments):
        top = segment.top
        if index > 0:
            top += interfaces[index - 1].half_support_km
        bottom = segment.bottom
        if index < len(interfaces):
            bottom -= interfaces[index].half_support_km
        if bottom > top and not np.isclose(bottom, top):
            regions.append((top, bottom, _segment_layer_count(segment.name, config)))
        if index < len(interfaces) and not interfaces[index].is_hard:
            support_top, support_bottom = interfaces[index].support_km
            regions.append((support_top, support_bottom, config.transition_layers))
    return regions


def _segment_layer_count(name: str | None, config: DiscretizationConfig) -> int | None:
    counts = config.layers_per_segment
    if counts is None:
        return None
    if isinstance(counts, Mapping):
        return int(counts[name]) if name in counts else None
    return int(counts)


def _layer_means(model, tops, bottoms, quadrature_points: int = 5):
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    z = 0.5 * (tops + bottoms)[:, None] + 0.5 * (bottoms - tops)[:, None] * nodes[None, :]
    return tuple(
        0.5 * np.sum(values.reshape(z.shape) * weights[None, :], axis=1)
        for values in model.evaluate_properties(z.ravel())
    )
