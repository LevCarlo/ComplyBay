"""Model ensembles and derived quantities from parameter samples.

Point values of a parameterization (such as the end values of a Vs gradient)
can trade off against each other while integral quantities stay well
constrained, so every segment also reports

- ``thickness``  (km),
- ``tp2``: vertical two-way P time, 2 * integral dz / Vp  (s),
- ``ts2``: vertical two-way S time, 2 * integral dz / Vs  (s),
- ``vs_mean``: travel-time average Vs, thickness / integral dz / Vs  (km/s),

integrated over the segment's depth interval of the blended model with
Gauss-Legendre quadrature; plus every interface depth and width.
"""

from __future__ import annotations

import numpy as np

from complybay.parameterization import ModelParameterization, ParameterizedModel

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(32)


def derived_quantities(model: ParameterizedModel) -> dict[str, float]:
    """Interface depths/widths and per-segment integral quantities of one model."""
    out: dict[str, float] = {}
    for index, interface in enumerate(model.interfaces):
        name = interface.name or f"interface{index}"
        out[f"depth.{name}"] = float(interface.depth_km)
        out[f"width.{name}"] = float(interface.width_km)
    for index, segment in enumerate(model.segments):
        name = segment.name or f"segment{index}"
        half = 0.5 * segment.thickness
        z = segment.top + half * (1.0 + _NODES)
        # Keep the bottom node inside the segment (its boundary belongs to the next one).
        z = np.minimum(z, segment.bottom - 1e-9 * segment.thickness)
        vs, vp, _ = model.evaluate_properties(z)
        s_time = half * np.sum(_WEIGHTS / vs)
        out[f"thickness.{name}"] = float(segment.thickness)
        out[f"tp2.{name}"] = float(2.0 * half * np.sum(_WEIGHTS / vp))
        out[f"ts2.{name}"] = float(2.0 * s_time)
        out[f"vs_mean.{name}"] = float(segment.thickness / s_time)
    return out


def model_ensemble(
    parameterization: ModelParameterization, samples, depths
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(vs, vp, rho) of shape (n_samples, n_depths) and derived quantities per sample."""
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    depths = np.asarray(depths, dtype=float)
    vs = np.empty((samples.shape[0], depths.size))
    vp = np.empty_like(vs)
    rho = np.empty_like(vs)
    derived: dict[str, list[float]] = {}
    for i, theta in enumerate(samples):
        model = parameterization.vector_to_model(theta)
        vs[i], vp[i], rho[i] = model.evaluate_properties(np.clip(depths, 0.0, model.zmax))
        for key, value in derived_quantities(model).items():
            derived.setdefault(key, []).append(value)
    return vs, vp, rho, {k: np.asarray(v) for k, v in derived.items()}
