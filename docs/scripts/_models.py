"""Synthetic models shared by the docs scripts (not a figure itself).

All values are assumed, chosen to look like a sedimented oceanic site; they
are not taken from a source.
"""

from __future__ import annotations

from complybay.model import LayeredModel, OBSModel, WaterLayer

WATER_DEPTH_KM = 2.0  # assumed

# Assumed: 0.6 km soft sediment, two crustal layers, mantle to 60 km.
SEDIMENT_OVER_CRUST = LayeredModel(
    thickness=[0.6, 2.0, 4.0, 53.4],
    vp=[1.8, 5.2, 6.7, 8.1, 8.1],
    vs=[0.3, 2.9, 3.75, 4.5, 4.5],
    rho=[1.8, 2.6, 2.9, 3.35, 3.35],
)


def sediment_over_crust(water_depth_km: float = WATER_DEPTH_KM) -> OBSModel:
    return OBSModel(SEDIMENT_OVER_CRUST, WaterLayer(water_depth_km))
