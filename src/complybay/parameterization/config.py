"""Build :class:`ParameterizedModel` objects from a ``Model`` mapping (YAML-like).

Adapted from SeisForge (``seisforge.inv.model_config``, commit 019d568). Any
numeric field may be a parameter reference ``{parameter: name}``, resolved
from the current parameter values::

    Model:
      water: {depth_km: 2.0, vp: 1.5, rho: 1.03}   # fixed; vp and rho optional
      zmax_km: 150.0                              # half-space top
      scaling: {vp: {method: brocher2005}, rho: {method: brocher2005}}
      interfaces:                                  # top to bottom
        - name: basement
          depth_km: {parameter: basement.depth}
          width_km: {parameter: basement.width}    # 10-90 % width, 0 = hard
          shape: sigmoid                           # or smootherstep
      segments:                                    # one more than interfaces
        - name: sediment
          vs: {type: gradient, top: {parameter: sed.vs_top}, bottom: {parameter: sed.vs_bot}}
          vp: {type: constant, value: {parameter: sed.vp}}   # optional; or vp_vs: {...}
          scaling: {rho: {method: gardner1974}}
          qs: {type: constant, value: {parameter: sed.qs}}   # optional; also qp
        - name: crust
          vs: {type: bspline, coefficients: [...], degree: 3}

Profiles: ``constant`` (value), ``gradient`` (top, bottom), ``bspline``
(coefficients, degree, knot_spacing, knot_alpha). Relation options other
than ``method`` are passed to the relation (``vp_vs``, ``rho``, ``a``, ``b``).
``qp`` / ``qs`` profiles make a segment attenuating (constant Q); then the
velocities are those at ``Model.q_reference_frequency_hz`` (default 1 Hz).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from complybay.model import WaterLayer
from complybay.parameterization.model import (
    BSplineProfile,
    ConstantProfile,
    GradientProfile,
    Interface,
    ParameterizedModel,
    Profile,
    Segment,
)
from complybay.parameterization.scaling import ScalingConfig


def resolve_value(value: Any, values: Mapping[str, float]) -> float:
    """A literal number, or ``{parameter: name}`` looked up in ``values``."""
    if isinstance(value, Mapping):
        if set(value) != {"parameter"}:
            raise ValueError("Parameter references must be {parameter: name}.")
        name = value["parameter"]
        if name not in values:
            raise ValueError(f"Unknown parameter reference: {name}")
        return float(values[name])
    return float(value)


def parameter_references(config: Any) -> set[str]:
    """All parameter names referenced anywhere in ``config``."""
    if isinstance(config, Mapping):
        if set(config) == {"parameter"}:
            return {str(config["parameter"])}
        return set().union(*(parameter_references(v) for v in config.values()))
    if isinstance(config, list | tuple):
        return set().union(*(parameter_references(v) for v in config))
    return set()


def validate_model_layout(model_cfg: Mapping[str, Any]) -> None:
    """Check the structure of a ``Model`` mapping without resolving values."""
    segments = model_cfg.get("segments")
    if not segments:
        raise ValueError("Model.segments must be a non-empty list.")
    if "zmax_km" not in model_cfg:
        raise ValueError("Model.zmax_km (top of the half-space) is required.")
    interfaces = model_cfg.get("interfaces") or []
    if len(segments) != len(interfaces) + 1:
        raise ValueError("Model.segments must have one more entry than Model.interfaces.")
    for index, item in enumerate(interfaces):
        if "depth_km" not in item:
            raise ValueError(f"Model.interfaces[{index}] needs depth_km.")
    for index, item in enumerate(segments):
        if "vs" not in item:
            raise ValueError(f"Model.segments[{index}] needs a vs profile.")
        if "top_km" in item or "bottom_km" in item:
            raise ValueError("Segment depths come from Model.interfaces; remove top_km/bottom_km.")


def model_from_config(
    model_cfg: Mapping[str, Any], values: Mapping[str, float] | None = None
) -> ParameterizedModel:
    """Build the continuous model, resolving parameter references from ``values``."""
    values = values or {}
    validate_model_layout(model_cfg)
    interfaces = tuple(
        Interface(
            depth_km=resolve_value(item["depth_km"], values),
            width_km=resolve_value(item.get("width_km", 0.0), values),
            shape=str(item.get("shape", "sigmoid")),
            name=item.get("name"),
        )
        for item in model_cfg.get("interfaces") or []
    )
    depths = [0.0, *(i.depth_km for i in interfaces), resolve_value(model_cfg["zmax_km"], values)]
    for upper, lower in zip(depths[:-1], depths[1:], strict=True):
        if lower <= upper:
            raise ValueError(
                f"Interface depths must increase downward ({upper:g} -> {lower:g} km)."
            )
    segments = tuple(
        Segment(
            top=depths[index],
            bottom=depths[index + 1],
            vs=profile_from_config(item["vs"], values),
            vp_vs=_optional_profile(item.get("vp_vs"), values),
            vp=_optional_profile(item.get("vp"), values),
            name=item.get("name"),
            scaling=None if item.get("scaling") is None else scaling_from_config(item["scaling"]),
            qp=_optional_profile(item.get("qp"), values),
            qs=_optional_profile(item.get("qs"), values),
        )
        for index, item in enumerate(model_cfg["segments"])
    )
    reference = model_cfg.get("q_reference_frequency_hz")
    return ParameterizedModel(
        segments=segments,
        interfaces=interfaces,
        scaling=scaling_from_config(model_cfg.get("scaling")),
        **({} if reference is None else {"q_reference_frequency": float(reference)}),
    )


def water_from_config(model_cfg: Mapping[str, Any]) -> WaterLayer | None:
    """The fixed water column, or ``None`` for a land station."""
    water = model_cfg.get("water")
    if water is None:
        return None
    return WaterLayer(
        thickness=float(water["depth_km"]),
        vp=float(water.get("vp", 1.5)),
        rho=float(water.get("rho", 1.03)),
    )


def profile_from_config(config: Mapping[str, Any], values: Mapping[str, float]) -> Profile:
    kind = str(config["type"]).lower()
    if kind == "constant":
        return ConstantProfile(value=resolve_value(config["value"], values))
    if kind == "gradient":
        return GradientProfile(
            top=resolve_value(config["top"], values),
            bottom=resolve_value(config["bottom"], values),
        )
    if kind == "bspline":
        return BSplineProfile(
            coefficients=tuple(resolve_value(c, values) for c in config["coefficients"]),
            degree=int(config.get("degree", 3)),
            knot_spacing=str(config.get("knot_spacing", "geometric")),
            knot_alpha=float(config.get("knot_alpha", 2.0)),
        )
    raise ValueError(f"Unknown profile type: {config['type']!r}")


def _optional_profile(config, values):
    return None if config is None else profile_from_config(config, values)


def scaling_from_config(config: Mapping[str, Any] | None) -> ScalingConfig:
    config = config or {}
    vp = dict(config.get("vp", {}))
    rho = dict(config.get("rho", {}))
    return ScalingConfig(
        vp=vp.pop("method", "brocher2005"),
        rho=rho.pop("method", "brocher2005"),
        vp_kwargs=vp,
        rho_kwargs=rho,
    )
