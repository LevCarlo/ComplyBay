"""Layered seafloor models.

Units are the same for every forward solver: thickness and depth in km,
velocities in km/s, density in g/cm^3. Depths below the seafloor start at 0.
Solvers that need other units (the compliance propagator works in m) convert
at their own boundary.

An :class:`OBSModel` is a water column (or none, for a land station) on top of
a :class:`LayeredModel`: ``n`` finite solid layers over a half-space, so the
property arrays have ``n + 1`` entries and ``thickness`` has ``n``.

Attenuation: optional ``qp`` and ``qs`` per layer (constant quality factors,
``inf`` = elastic). With them, ``vp`` and ``vs`` are phase velocities at
``q_reference_frequency`` (Hz) and every solver disperses them to its own
frequencies (:mod:`complybay.forward.attenuation`). Without them the model is
elastic and the velocities do not depend on frequency. The water column is
elastic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

Property = Literal["vp", "vs", "rho"]
# Assumed: velocities of an attenuating model are those at 1 Hz, the usual
# reference frequency of seismic velocity models.
Q_REFERENCE_FREQUENCY = 1.0
PROPERTIES: tuple[Property, ...] = ("vp", "vs", "rho")


def _readonly(values, name: str, *, ndim: int = 1) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class WaterLayer:
    """Homogeneous water column above the seafloor."""

    thickness: float
    vp: float = 1.5
    rho: float = 1.03

    def __post_init__(self):
        for name in ("thickness", "vp", "rho"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"WaterLayer.{name} must be positive, got {value}.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LayeredModel:
    """Isotropic elastic layers over a half-space, top to bottom."""

    thickness: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    qp: np.ndarray | None = None  # None: elastic
    qs: np.ndarray | None = None
    q_reference_frequency: float = Q_REFERENCE_FREQUENCY

    def __post_init__(self):
        thickness = _readonly(self.thickness, "thickness")
        props = {name: _readonly(getattr(self, name), name) for name in PROPERTIES}
        n = thickness.size
        for name, values in props.items():
            if values.size != n + 1:
                raise ValueError(
                    f"{name} needs {n + 1} values ({n} layers and the half-space), "
                    f"got {values.size}."
                )
        if np.any(thickness <= 0):
            raise ValueError("Layer thicknesses must be positive.")
        if np.any(props["vs"] <= 0) or np.any(props["rho"] <= 0):
            raise ValueError("vs and rho must be positive; put water in a WaterLayer.")
        # Positive bulk modulus: lambda + 2/3 mu > 0.
        if np.any(props["vp"] ** 2 <= 4.0 / 3.0 * props["vs"] ** 2):
            raise ValueError("vp must exceed sqrt(4/3) * vs in every layer.")
        object.__setattr__(self, "thickness", thickness)
        for name, values in props.items():
            object.__setattr__(self, name, values)
        for name in ("qp", "qs"):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.array(values, dtype=float)
            if values.shape != (n + 1,) or np.any(np.isnan(values)) or np.any(values <= 0):
                raise ValueError(f"{name} needs {n + 1} positive values (inf = elastic).")
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        if not np.isfinite(self.q_reference_frequency) or self.q_reference_frequency <= 0:
            raise ValueError("q_reference_frequency must be positive.")

    @property
    def attenuating(self) -> bool:
        """True if any layer has a finite qp or qs."""
        return any(q is not None and np.any(np.isfinite(q)) for q in (self.qp, self.qs))

    def q(self, name: Literal["qp", "qs"]) -> np.ndarray:
        """Quality factors per layer, ``inf`` where elastic or not given."""
        values = getattr(self, name)
        return np.full(self.n_layers + 1, np.inf) if values is None else values

    @classmethod
    def from_legacy_array(cls, model) -> LayeredModel:
        """Build from the v0.1 array: rows (thickness m, rho, vp, vs), last row half-space."""
        model = np.asarray(model, dtype=float)
        if model.ndim != 2 or model.shape[1] != 4 or model.shape[0] < 1:
            raise ValueError("Legacy model must have shape (n_layers + 1, 4).")
        return cls(
            thickness=model[:-1, 0] * 1e-3,
            vp=model[:, 2],
            vs=model[:, 3],
            rho=model[:, 1],
        )

    @property
    def n_layers(self) -> int:
        """Number of finite layers (the half-space not included)."""
        return self.thickness.size

    @property
    def tops(self) -> np.ndarray:
        """Depth of the top of each layer and of the half-space (km)."""
        return np.concatenate(([0.0], np.cumsum(self.thickness)))

    def property(self, name: Property) -> np.ndarray:
        if name not in PROPERTIES:
            raise ValueError(f"Unknown property {name!r}; choose from {PROPERTIES}.")
        return getattr(self, name)

    def with_property(self, name: Property, values) -> LayeredModel:
        """Return a copy with one property replaced."""
        self.property(name)
        return replace(self, **{name: values})

    def scaled(self, name: Property, index: int, factor: float) -> LayeredModel:
        """Return a copy with ``name`` of layer ``index`` multiplied by ``factor``."""
        values = self.property(name).copy()
        values[index] *= factor
        return self.with_property(name, values)

    def resampled(self, dz: float) -> LayeredModel:
        """Split every layer thicker than ``dz`` into equal sublayers no thicker than ``dz``."""
        if dz <= 0:
            raise ValueError("dz must be positive.")
        counts = np.maximum(np.ceil(self.thickness / dz - 1e-9).astype(int), 1)
        thickness = np.repeat(self.thickness / counts, counts)
        repeat = np.append(counts, 1)
        return replace(
            self,
            thickness=thickness,
            vp=np.repeat(self.vp, repeat),
            vs=np.repeat(self.vs, repeat),
            rho=np.repeat(self.rho, repeat),
            qp=None if self.qp is None else np.repeat(self.qp, repeat),
            qs=None if self.qs is None else np.repeat(self.qs, repeat),
        )


@dataclass(frozen=True)
class OBSModel:
    """A station on the seafloor: optional water column over a layered solid."""

    solid: LayeredModel
    water: WaterLayer | None = field(default=None)

    @property
    def water_depth(self) -> float:
        """Water depth in km (0 for a land station)."""
        return 0.0 if self.water is None else self.water.thickness

    def with_solid(self, solid: LayeredModel) -> OBSModel:
        return replace(self, solid=solid)
