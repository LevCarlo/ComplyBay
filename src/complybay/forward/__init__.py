"""Forward solvers for seafloor observables.

All solvers take an :class:`complybay.model.OBSModel` (km, km/s, g/cm^3).
The modules ``ncomp_fortran``, ``k_solver``, ``complykernel`` and
``ncomp_matlab_py`` are the v0.1 interface, kept for the legacy inversion code.
"""

from complybay.forward.acf import synthetic_acf, synthetic_acfs, wraparound_fraction
from complybay.forward.compliance import compliance, compliance_kernel, water_wavenumber
from complybay.forward.dispersion import phase_velocity, phase_velocity_kernel
from complybay.forward.response import SeafloorResponse, seafloor_response

__all__ = [
    "SeafloorResponse",
    "compliance",
    "compliance_kernel",
    "phase_velocity",
    "phase_velocity_kernel",
    "seafloor_response",
    "synthetic_acf",
    "synthetic_acfs",
    "water_wavenumber",
    "wraparound_fraction",
]
