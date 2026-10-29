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

Attenuating layers (finite ``qp`` or ``qs``) take the complex Kjartansson
velocities of :mod:`complybay.forward.attenuation`, which depend on frequency,
so their eigenvector matrices are built per frequency; elastic layers keep the
single frequency-independent matrix. The vertical slowness is the principal
root, whose imaginary part is positive for complex velocities with
attenuation, so downgoing waves decay with depth. The zero-frequency sample,
where constant-Q velocities are undefined, is computed with the reference
velocities; every output is zero there (all carry a factor omega).

Sign conventions of the outputs: ``vertical`` is positive up, ``radial`` is
positive in the direction of horizontal propagation, ``pressure`` is positive
in compression. The incident wave has unit displacement amplitude and moves
the ground up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from complybay.forward.attenuation import complex_velocity
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


def _solid_eigenvectors_per_frequency(vp, vs, rho: float, p: float):
    """As :func:`_solid_eigenvectors` for complex velocities ``vp``, ``vs`` (n_f,).

    Returns eigenvectors (n_f, 4, 4) and vertical slownesses (n_f, 4).
    """
    mu = rho * vs**2
    lam = rho * vp**2 - 2 * mu
    qa = np.sqrt(1.0 / vp**2 - p**2)
    qb = np.sqrt(1.0 / vs**2 - p**2)
    columns, slowness = [], []
    for kind, q in (("P", qa), ("S", qb), ("P", -qa), ("S", -qb)):
        if kind == "P":
            ux, uz = p * vp, q * vp
        else:
            ux, uz = q * vs, -p * vs
        txz = mu * (p * uz + q * ux)
        tzz = lam * (p * ux + q * uz) + 2 * mu * q * uz
        columns.append(np.stack((ux, uz, txz, tzz), axis=-1))
        slowness.append(q)
    return np.stack(columns, axis=-1), np.stack(slowness, axis=-1)


def _fluid_eigenvectors(vp: float, rho: float, p: float):
    """Columns (u_z, tau_zz) for downgoing and upgoing P in a fluid."""
    q = _vertical_slowness(vp, p)
    matrix = np.array([[q * vp, -q * vp], [rho * vp, rho * vp]], dtype=complex)
    return matrix, np.array([q, -q])


def _propagate(eigvecs, slowness, omega, thickness, state):
    """Carry ``state`` (m, n_f, k) through a layer of ``thickness`` for every omega.

    Equivalent to E diag(exp(i omega q h)) E^-1 @ state[:, f, :] for each f.
    The frequency axis is folded into the columns so each product is one
    (m, m) @ (m, n_f * k) matrix multiplication.
    """
    m, n, k = state.shape
    phase = np.exp(1j * slowness[:, None] * omega[None, :] * thickness)  # (m, n_f)
    x = (np.linalg.inv(eigvecs) @ state.reshape(m, n * k)).reshape(m, n, k)
    x *= phase[:, :, None]
    return (eigvecs @ x.reshape(m, n * k)).reshape(m, n, k)


def _propagate_per_frequency(eigvecs, slowness, omega, thickness, state):
    """As :func:`_propagate` with eigenvectors (n_f, m, m) and slownesses (n_f, m)."""
    x = np.linalg.solve(eigvecs, state.transpose(1, 0, 2))  # (n_f, m, k)
    x *= np.exp(1j * slowness * omega[:, None] * thickness)[:, :, None]
    return (eigvecs @ x).transpose(1, 0, 2)


def _layer_matrices(solid, index: int, ray_p: float, freqs: np.ndarray):
    """(eigenvectors, slownesses, per_frequency) of solid layer ``index``."""
    vp, vs, rho = solid.vp[index], solid.vs[index], solid.rho[index]
    qp, qs = solid.q("qp")[index], solid.q("qs")[index]
    if np.isinf(qp) and np.isinf(qs):
        e, q = _solid_eigenvectors(vp, vs, rho, ray_p)
        return e, q, False
    f_ref = solid.q_reference_frequency
    positive = np.where(freqs > 0, freqs, f_ref)  # f = 0: reference velocities
    cvp = np.where(freqs > 0, complex_velocity(vp, qp, positive, f_ref), vp)
    cvs = np.where(freqs > 0, complex_velocity(vs, qs, positive, f_ref), vs)
    e, q = _solid_eigenvectors_per_frequency(cvp, cvs, rho, ray_p)
    return e, q, True


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

    # Unknowns x = (u_x, u_z, tau_zz) at the seafloor; tau_xz(0) = 0 (fluid or vacuum
    # above). Carry the three seafloor basis vectors to the half-space top, where
    # m[f] maps x to the half-space wave amplitudes (P down, S down, P up, S up).
    n = freqs.size
    # State layout (vector component, frequency, basis vector).
    basis = np.repeat(np.eye(4, dtype=complex)[:, None, [0, 1, 3]], n, axis=1)
    for index, h in enumerate(solid.thickness):
        e, q, per_frequency = _layer_matrices(solid, index, ray_p, freqs)
        propagate = _propagate_per_frequency if per_frequency else _propagate
        basis = propagate(e, q, omega, h, basis)
    e_hs, _, per_frequency = _layer_matrices(solid, solid.n_layers, ray_p, freqs)
    if per_frequency:
        m = np.linalg.solve(e_hs, basis.transpose(1, 0, 2))
    else:
        m = (np.linalg.inv(e_hs) @ basis.reshape(4, n * 3)).reshape(4, n, 3).transpose(1, 0, 2)

    lhs = np.zeros((n, 3, 3), dtype=complex)
    rhs = np.zeros((n, 3), dtype=complex)
    lhs[:, :2, :] = m[:, 2:, :]
    # Unit incident upgoing P; its polarization (p, -q) * vp points up (z is down).
    rhs[:, 0] = 1.0
    if model.water is None:
        lhs[:, 2, 2] = 1.0  # free surface: tau_zz(0) = 0
    else:
        w = model.water
        ew, qw = _fluid_eigenvectors(w.vp, w.rho, ray_p)
        # Seafloor (u_z, tau_zz) basis carried to the sea surface, where tau_zz vanishes.
        pw = _propagate(
            ew, qw, omega, -w.thickness, np.repeat(np.eye(2, dtype=complex)[:, None], n, axis=1)
        )
        lhs[:, 2, 1] = pw[1, :, 0]
        lhs[:, 2, 2] = pw[1, :, 1]
    x = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    ux, uz, tzz = x[:, 0], x[:, 1], x[:, 2]
    amplitudes = (m @ x[..., None])[..., 0]

    return SeafloorResponse(
        freqs=freqs,
        ray_p=float(ray_p),
        displacement_z=-uz,
        displacement_x=ux,
        pressure=-(1j * omega) * tzz,
        reflected_p=amplitudes[:, 0],
        reflected_s=amplitudes[:, 1],
    )
