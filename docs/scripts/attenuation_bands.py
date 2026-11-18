"""Figure attenuation_bands.png: constant-Q velocity dispersion across the data bands.

(a) Phase velocity relative to the reference velocity, c(f) / v_ref, for
f_ref = 1 Hz and several Q (package: ``phase_velocity_at`` as lines and the
phase velocity omega / Re k of ``complex_velocity`` as dots), with the bands of
the three data types shaded. (b) Compliance of the shared docs model with
Qs = 20, Qp = 100 in the sediment, relative to the elastic model, for three
reference frequencies.

Run: python docs/scripts/attenuation_bands.py
"""

from __future__ import annotations

from dataclasses import replace

import _style
import numpy as np
from _models import sediment_over_crust
from _style import plt

from complybay.acf import preset
from complybay.forward import compliance
from complybay.forward.attenuation import complex_velocity, phase_velocity_at
from complybay.forward.compliance import infragravity_cutoff

F_REF = 1.0  # Hz
QS = (10.0, 20.0, 50.0, 100.0)


def phase_ratio(q, freqs, f_ref):
    """c(f) / v_ref from the package (phase_velocity_at takes one frequency)."""
    return np.array([phase_velocity_at(1.0, q, f, f_ref) for f in freqs])


# Bands: compliance up to f_c of 2 km of water; Rayleigh 8-30 s; kim2023 half-power band.
f_grid = np.geomspace(1e-3, 10.0, 400)
kim = preset("kim2023")
response = np.prod([f.power_response(f_grid) for f in kim.filters], axis=0)
acf_band = f_grid[response >= 0.5 * response.max()]
bands = {
    "compliance": (0.003, infragravity_cutoff(2000.0)),
    "Rayleigh": (1 / 30, 1 / 8),
    "ACF": (acf_band[0], acf_band[-1]),
}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
for (name, (lo, hi)), color in zip(bands.items(), ("0.85", "0.75", "0.85"), strict=True):
    ax_a.axvspan(lo, hi, color=color, lw=0)
    ax_a.text(np.sqrt(lo * hi), 1.105, name, ha="center", va="top", fontsize=7)
for q, color in zip(QS, _style.SERIES, strict=False):
    ax_a.plot(f_grid, phase_ratio(q, f_grid, F_REF), color=color, lw=1.2, label=f"Q = {q:g}")
    dots = f_grid[::40]
    phase = 1.0 / (1.0 / complex_velocity(1.0, q, dots, F_REF)).real
    assert np.allclose(phase, phase_ratio(q, dots, F_REF), rtol=1e-12)
    ax_a.plot(dots, phase, "o", ms=3, color=color)
ax_a.axvline(F_REF, color=_style.TRUTH, lw=0.6, ls=":")
ax_a.set_xscale("log")
ax_a.set_ylim(0.84, 1.11)
ax_a.set_xlabel("frequency (Hz)")
ax_a.set_ylabel(r"$c(f)\,/\,v_\mathrm{ref}$")
ax_a.set_title(r"$f_\mathrm{ref}$ = 1 Hz (dotted)")
ax_a.legend(loc="lower right")
_style.panel_label(ax_a, "(a)")

# (b) Compliance of the shared model, sediment Qs = 20, Qp = 100.
model = sediment_over_crust()
solid = model.solid
freqs = np.geomspace(0.003, infragravity_cutoff(2000.0), 25)
eta_elastic = compliance(model, freqs)
q_s = np.r_[20.0, np.full(solid.n_layers, np.inf)]
q_p = np.r_[100.0, np.full(solid.n_layers, np.inf)]
for f_ref, color in zip((1.0, 0.1, 0.01), _style.SERIES, strict=False):
    attenuating = model.with_solid(replace(solid, qs=q_s, qp=q_p, q_reference_frequency=f_ref))
    ratio = compliance(attenuating, freqs) / eta_elastic
    ax_b.plot(freqs, ratio, "o", ms=3, color=color, label=rf"$f_\mathrm{{ref}}$ = {f_ref:g} Hz")
    # Below f_ref the sediment is slower, hence more compliant. At f = f_ref the
    # dispersed model is the elastic one, so eta is the same (round-off, 1e-12).
    assert np.all(ratio[freqs < f_ref] > 1)
    if f_ref <= freqs.max():
        same = compliance(attenuating, [f_ref]) / compliance(model, [f_ref])
        assert abs(same[0] - 1) < 1e-12, same
ax_b.set_xscale("log")
ax_b.set_xlabel("frequency (Hz)")
ax_b.set_ylabel(r"$\eta_Q / \eta_\mathrm{elastic}$")
ax_b.set_title(r"sediment $Q_s$ = 20, $Q_p$ = 100")
ax_b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
_style.panel_label(ax_b, "(b)")

fig.tight_layout()
_style.save(fig, __file__)
