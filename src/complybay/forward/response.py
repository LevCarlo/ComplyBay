"""Seafloor response to a teleseismic plane P wave.

A plane P wave with horizontal slowness ``ray_p`` (s/km) arrives from the
half-space below a stack of elastic layers, an optional water column and the
sea surface (pressure-free). The seafloor records the vertical and radial
motion of the solid and the pressure in the water just above it. Unlike a
land station the receiver is not on the free surface, so the water column
adds its own reverberations and pressure and vertical motion see them with
different signs.

Method: frequency-domain propagator in the motion-stress vector
``v = (u_x, u_z, tau_xz, tau_zz)`` with z positive down, tractions scaled as
``tau = sigma / (i omega)`` so the layer eigenvector matrices do not depend on
frequency. Time dependence is exp(-i omega t). All waves are taken as
propagating or decaying; teleseismic slownesses keep every layer except
possibly a very fast half-space propagating.

Sign conventions of the outputs: ``vertical`` is positive up, ``radial`` is
positive in the direction of horizontal propagation, ``pressure`` is positive
in compression. The incident wave has unit displacement amplitude and moves
the ground up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from complybay.model import OBSModel


def _vertical_slowness(v: float, p: float) -> complex:
    # Principal sqrt keeps Im >= 0, so a downgoing evanescent wave decays with depth.
    return np.sqrt(complex(1.0 / v**2 - p**2))


def _solid_eigenvectors(vp: float, vs: float, rho: float, p: float):
    """Columns: downgoing P, downgoing S, upgoing P, upgoing S (unit displacement)."""
    mu = rho * vs**2
    lam = rho * vp**2 - 2 * mu
    qa = _vertical_slowness(vp, p)
    qb = _vertical_slowness(vs, p)
    columns, slowness = [], []
    for kind, q in (("P", qa), ("S", qb), ("P", -qa), ("S", -qb)):
        if kind == "P":
            ux, uz = p * vp, q * vp
        else:
            ux, uz = q * vs, -p * vs
        txz = mu * (p * uz + q * ux)
        tzz = lam * (p * ux + q * uz) + 2 * mu * q * uz
        columns.append((ux, uz, txz, tzz))
        slowness.append(q)
    return np.array(columns, dtype=complex).T, np.array(slowness, dtype=complex)


def _fluid_eigenvectors(vp: float, rho: float, p: float):
    """Columns (u_z, tau_zz) for downgoing and upgoing P in a fluid."""
    q = _vertical_slowness(vp, p)
    matrix = np.array([[q * vp, -q * vp], [rho * vp, rho * vp]], dtype=complex)
    return matrix, np.array([q, -q])


def _propagate(eigvecs, slowness, omega, thickness):
    """Batched layer propagators exp over ``thickness`` for every omega: (n_f, m, m)."""
    phase = np.exp(1j * omega[:, None] * slowness[None, :] * thickness)
    return np.einsum("ij,fj,jk->fik", eigvecs, phase, np.linalg.inv(eigvecs))


@dataclass(frozen=True)
class SeafloorResponse:
    """Seafloor transfer functions for a unit incident P wave."""

    freqs: np.ndarray
    ray_p: float
    displacement_z: np.ndarray  # positive up
    displacement_x: np.ndarray
    pressure: np.ndarray  # positive in compression; zero for a land station
    reflected_p: np.ndarray  # downgoing P amplitude in the half-space
    reflected_s: np.ndarray  # downgoing S amplitude in the half-space

    def component(self, name: str) -> np.ndarray:
        """Spectrum of ``vertical`` or ``radial`` ground velocity, or ``pressure``."""
        minus_i_omega = -2j * np.pi * self.freqs
        if name == "vertical":
            return minus_i_omega * self.displacement_z
        if name == "radial":
            return minus_i_omega * self.displacement_x
        if name == "pressure":
            return self.pressure
        raise ValueError(f"Unknown component {name!r}; use vertical, radial or pressure.")


def seafloor_response(model: OBSModel, ray_p: float, freqs) -> SeafloorResponse:
    """Transfer functions at the seafloor for frequencies ``freqs`` (Hz)."""
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if ray_p < 0:
        raise ValueError("ray_p must be non-negative.")
    omega = 2 * np.pi * freqs
    solid = model.solid

    # Seafloor to half-space top: A maps v(0) to v(z_hs).
    a = np.broadcast_to(np.eye(4, dtype=complex), (freqs.size, 4, 4)).copy()
    for h, vp, vs, rho in zip(solid.thickness, solid.vp, solid.vs, solid.rho, strict=False):
        e, q = _solid_eigenvectors(vp, vs, rho, ray_p)
        a = _propagate(e, q, omega, h) @ a
    e_hs, _ = _solid_eigenvectors(solid.vp[-1], solid.vs[-1], solid.rho[-1], ray_p)
    m = np.linalg.inv(e_hs)[None] @ a  # v(0) -> half-space wave amplitudes

    # Unknowns x = (u_x, u_z, tau_zz) at the seafloor; tau_xz(0) = 0 (fluid or vacuum above).
    n = freqs.size
    lhs = np.zeros((n, 3, 3), dtype=complex)
    rhs = np.zeros((n, 3), dtype=complex)
    lhs[:, :2, :] = m[:, 2:, :][:, :, [0, 1, 3]]
    # Unit incident upgoing P; its polarization (p, -q) * vp points up (z is down).
    rhs[:, 0] = 1.0
    if model.water is None:
        lhs[:, 2, 2] = 1.0  # free surface: tau_zz(0) = 0
    else:
        w = model.water
        ew, qw = _fluid_eigenvectors(w.vp, w.rho, ray_p)
        pw = _propagate(ew, qw, omega, -w.thickness)  # seafloor -> sea surface
        lhs[:, 2, 1] = pw[:, 1, 0]
        lhs[:, 2, 2] = pw[:, 1, 1]  # tau_zz at the sea surface vanishes
    x = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    ux, uz, tzz = x[:, 0], x[:, 1], x[:, 2]
    v0 = np.stack((ux, uz, np.zeros_like(ux), tzz), axis=1)
    amplitudes = np.einsum("fij,fj->fi", m, v0)

    return SeafloorResponse(
        freqs=freqs,
        ray_p=float(ray_p),
        displacement_z=-uz,
        displacement_x=ux,
        pressure=-(1j * omega) * tzz,
        reflected_p=amplitudes[:, 0],
        reflected_s=amplitudes[:, 1],
    )
