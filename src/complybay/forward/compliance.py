"""Seafloor compliance forward model.

Ocean surface gravity (infragravity) waves load the seafloor with a pressure
field of horizontal wavenumber k, fixed by the water depth H through

    omega^2 = g k tanh(k H).

The normalized compliance is the ratio of vertical seafloor displacement to
that pressure, scaled by k (Crawford et al., 1991):

    eta(omega) = k u_z / p_seafloor  =  -k v_z(0) / (omega sigma_zz(0))  [1/Pa]

The solid response comes from Crawford's propagator (``raydep_ft``, Fortran,
via MDN-Compliance) evaluated at horizontal slowness k / omega. The water
column enters only through k: compliance uses the water depth, not the water
velocity or density.

Attenuating models (finite qp/qs): each frequency uses the elastic model with
the layers' phase velocities at that frequency
(:func:`complybay.forward.attenuation.dispersed`); the imaginary part of the
moduli changes |eta| only at order 1/Q^2 and is left out.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from complybay.forward import raydep_ft
from complybay.forward.attenuation import dispersed
from complybay.forward.k_solver import solve_k_poly
from complybay.model import OBSModel
from complybay.model.layered import Property

G_DEFAULT = 9.79329  # m/s^2, the value used by Crawford's code
WavenumberMethod = Literal["exact", "poly"]


def water_wavenumber(
    omega,
    depth_m: float,
    *,
    method: WavenumberMethod = "exact",
    g: float = G_DEFAULT,
    rtol: float = 1e-14,
    max_iter: int = 50,
) -> np.ndarray:
    """Wavenumber (1/m) of surface gravity waves in water of depth ``depth_m`` (m).

    ``method="exact"`` solves omega^2 = g k tanh(k H) by vectorized Newton
    iteration from Eckart's approximation. ``method="poly"`` is the series
    approximation used by v0.1 (MDN-Compliance ``gravd``); it is up to ~2 %
    off the exact root near k H ~ 1-2 (up to ~3 % in compliance) and is kept
    only to reproduce old results.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if depth_m <= 0:
        raise ValueError("Water depth must be positive.")
    if np.any(omega < 0):
        raise ValueError("Angular frequencies must be non-negative.")
    if method == "poly":
        return np.atleast_1d(solve_k_poly(omega, depth_m, g))
    if method != "exact":
        raise ValueError(f"Unknown wavenumber method {method!r}.")

    k = np.zeros_like(omega)
    live = omega > 0
    w2 = omega[live] ** 2
    k_deep = w2 / g
    # Eckart (1951): k ~ k_deep / sqrt(tanh(k_deep H)), within a few percent everywhere.
    kk = k_deep / np.sqrt(np.tanh(k_deep * depth_m))
    for _ in range(max_iter):
        t = np.tanh(kk * depth_m)
        f = g * kk * t - w2
        df = g * (t + kk * depth_m * (1.0 - t**2))
        step = f / df
        kk = kk - step
        if np.all(np.abs(step) <= rtol * kk):
            break
    else:
        raise RuntimeError("Water wavenumber iteration did not converge.")
    k[live] = kk
    return k


def infragravity_cutoff(depth_m: float, *, g: float = G_DEFAULT) -> float:
    """Frequency (Hz) at which the gravity-wave wavelength equals the water depth.

    With k = 2 pi / H in omega^2 = g k tanh(k H):
    f_c = sqrt(g tanh(2 pi) / (2 pi H)), within 1e-5 of the deep-water
    sqrt(g / (2 pi H)). Linear wave theory attenuates the wave pressure from
    the sea surface to the seafloor by 1 / cosh(k H), which is 1/268 at f_c and
    falls off exponentially above it, so compliance measured above f_c is
    dominated by noise; f_c is the usual upper frequency limit of compliance
    data (0.028 Hz in 2 km of water).
    """
    if depth_m <= 0:
        raise ValueError("Water depth must be positive.")
    return float(np.sqrt(g * np.tanh(2 * np.pi) / (2 * np.pi * depth_m)))


def _raydep_array(model: OBSModel) -> np.ndarray:
    """Rows (thickness m, rho, vp, vs) as raydep_ft expects; half-space thickness unused."""
    solid = model.solid
    thickness_m = np.append(solid.thickness * 1e3, 0.0)
    return np.asfortranarray(np.column_stack((thickness_m, solid.rho, solid.vp, solid.vs)))


def compliance(
    model: OBSModel,
    freqs,
    *,
    wavenumber: WavenumberMethod = "exact",
    g: float = G_DEFAULT,
    decay_cutoff: float = 25.0,
) -> np.ndarray:
    """Normalized compliance (1/Pa) at frequencies ``freqs`` (Hz).

    The loading decays roughly as exp(-k z) below the seafloor, and the
    propagator overflows when it integrates that growth over a deep model. For
    each frequency the model is therefore cut at k z = ``decay_cutoff``: the
    layer that starts below that depth becomes the half-space. With the
    default the neglected structure sits below an e^-25 decay.
    """
    if model.water is None:
        raise ValueError("Compliance needs a water layer.")
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if np.any(freqs <= 0):
        raise ValueError("Frequencies must be positive.")
    if model.solid.attenuating:
        settings = dict(wavenumber=wavenumber, g=g, decay_cutoff=decay_cutoff)
        return np.array(
            [
                compliance(model.with_solid(dispersed(model.solid, f)), [f], **settings)[0]
                for f in freqs
            ]
        )
    omega = 2 * np.pi * freqs
    k = water_wavenumber(omega, model.water_depth * 1e3, method=wavenumber, g=g)
    rows = _raydep_array(model)

    # Layers kept per frequency: those whose top lies above the cut depth.
    tops_m = model.solid.tops[:-1] * 1e3
    n_keep = np.searchsorted(tops_m, decay_cutoff / k, side="left")
    eta = np.empty(freqs.size)
    for n in np.unique(n_keep):
        sel = n_keep == n
        sub = np.asfortranarray(rows[: n + 1])
        v, _u, zz, _zx = raydep_ft.raydep_batch_ft(
            k[sel] / omega[sel], omega[sel], sub, sub.shape[0], int(sel.sum())
        )
        eta[sel] = -k[sel] * v[0, :] / (omega[sel] * zz[0, :])
    return eta


def compliance_kernel(
    model: OBSModel,
    freqs,
    parameter: Property = "vs",
    *,
    rel_step: float = 0.025,
    wavenumber: WavenumberMethod = "exact",
    g: float = G_DEFAULT,
) -> np.ndarray:
    """Relative sensitivity d ln(eta) / d ln(m) of every layer and the half-space.

    Central differences with relative perturbation ``rel_step``. Returns shape
    ``(n_freqs, n_layers + 1)``; the last column is the half-space. Kernels
    depend on layer thickness, so resample the model (``LayeredModel.resampled``)
    to compare layers on an even depth grid.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    kwargs = {"wavenumber": wavenumber, "g": g}
    eta0 = compliance(model, freqs, **kwargs)
    solid = model.solid
    kernel = np.empty((freqs.size, solid.n_layers + 1))
    for i in range(solid.n_layers + 1):
        plus = compliance(
            model.with_solid(solid.scaled(parameter, i, 1 + rel_step)), freqs, **kwargs
        )
        minus = compliance(
            model.with_solid(solid.scaled(parameter, i, 1 - rel_step)), freqs, **kwargs
        )
        kernel[:, i] = (plus - minus) / (2 * rel_step * eta0)
    return kernel


def halfspace_compliance(vp: float, vs: float, rho: float) -> float:
    """Static normalized compliance of a uniform half-space (1/Pa), units km/s and g/cm^3.

    eta = (lambda + 2 mu) / (2 mu (lambda + mu)); the low-frequency limit of
    :func:`compliance` for a uniform solid.
    """
    mu = rho * 1e3 * (vs * 1e3) ** 2
    lam = rho * 1e3 * (vp * 1e3) ** 2 - 2 * mu
    return (lam + 2 * mu) / (2 * mu * (lam + mu))
