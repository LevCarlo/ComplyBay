"""Empirical Vp and density relations for segments whose Vp/rho are not sampled.

Vp relations take Vs (km/s); density relations take Vp (km/s), so density
always follows the Vp actually used in the segment, whether that Vp comes from
a relation or from a sampled Vp/Vs profile.

Vp from Vs:

- ``brocher2005``: Brocher (2005, BSSA 95, 2081-2092), Eq. 9, valid for
  0 <= Vs <= 4.5 km/s.
- ``castagna1985``: mudrock line Vp = 1.16 Vs + 1.36 (Castagna et al., 1985,
  Geophysics 50, 571-581); called "Castegna1995" in ComplyBay v0.1.
- ``constant_vp_vs``: Vp = ``vp_vs`` * Vs.

Density from Vp:

- ``brocher2005``: Brocher's (2005, Eq. 1) polynomial fit to the Nafe-Drake
  curve, valid for 1.5 <= Vp <= 8.5 km/s.
- ``gardner1974``: rho = a Vp^b with a = 1.741, b = 0.25 (Gardner et al., 1974).
- ``constant``: rho = ``rho``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def brocher_vp_from_vs(vs) -> np.ndarray:
    vs = np.asarray(vs, dtype=float)
    if np.any((vs < 0.0) | (vs > 4.5)):
        raise ValueError("Brocher (2005) Vp(Vs) is only valid for 0 <= Vs <= 4.5 km/s.")
    return 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4


def castagna_vp_from_vs(vs) -> np.ndarray:
    return 1.16 * np.asarray(vs, dtype=float) + 1.36


def constant_vp_vs(vs, *, vp_vs: float) -> np.ndarray:
    return vp_vs * np.asarray(vs, dtype=float)


def brocher_rho_from_vp(vp) -> np.ndarray:
    vp = np.asarray(vp, dtype=float)
    if np.any((vp < 1.5) | (vp > 8.5)):
        raise ValueError("Brocher (2005) rho(Vp) is only valid for 1.5 <= Vp <= 8.5 km/s.")
    return 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5


def gardner_rho_from_vp(vp, *, a: float = 1.741, b: float = 0.25) -> np.ndarray:
    return a * np.asarray(vp, dtype=float) ** b


def constant_rho(vp, *, rho: float) -> np.ndarray:
    return np.full_like(np.asarray(vp, dtype=float), rho)


VP_RELATIONS: dict[str, Callable[..., np.ndarray]] = {
    "brocher2005": brocher_vp_from_vs,
    "castagna1985": castagna_vp_from_vs,
    "constant_vp_vs": constant_vp_vs,
}
RHO_RELATIONS: dict[str, Callable[..., np.ndarray]] = {
    "brocher2005": brocher_rho_from_vp,
    "gardner1974": gardner_rho_from_vp,
    "constant": constant_rho,
}


@dataclass(frozen=True)
class ScalingConfig:
    """Which relations give Vp (from Vs) and density (from Vp) in a segment."""

    vp: str = "brocher2005"
    rho: str = "brocher2005"
    vp_kwargs: Mapping[str, Any] = field(default_factory=dict)
    rho_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.vp not in VP_RELATIONS:
            raise ValueError(
                f"Unknown Vp relation {self.vp!r}; choose from {sorted(VP_RELATIONS)}."
            )
        if self.rho not in RHO_RELATIONS:
            raise ValueError(
                f"Unknown density relation {self.rho!r}; choose from {sorted(RHO_RELATIONS)}."
            )
        object.__setattr__(self, "vp_kwargs", dict(self.vp_kwargs))
        object.__setattr__(self, "rho_kwargs", dict(self.rho_kwargs))

    def vp_from_vs(self, vs) -> np.ndarray:
        return VP_RELATIONS[self.vp](vs, **self.vp_kwargs)

    def rho_from_vp(self, vp) -> np.ndarray:
        return RHO_RELATIONS[self.rho](vp, **self.rho_kwargs)
