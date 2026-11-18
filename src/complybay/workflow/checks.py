"""Checks of configured observations against the physics of the data.

Run at setup, before sampling. They never raise: each returns notes, at level
``"warning"`` (likely a mistake or unusable data) or ``"info"`` (numbers worth
knowing before reading the results). The formulas are in the forward modules.

- warning: compliance frequencies above the infragravity limit
  f_c = sqrt(g tanh(2 pi) / (2 pi H))
  (:func:`complybay.forward.infragravity_cutoff`), where the wavelength
  equals the water depth H (0.028 Hz in 2 km of water): the wave pressure at
  the seafloor is attenuated by more than cosh(2 pi) = 268, so the data are
  noise.
- warning: an ACF window ending beyond the data's last lag.
- warning: an ACF window that includes the normalizing zero lag
  (``normalize: zero_lag``): fixed to 1 in data and synthetics, so it carries
  no information but counts in N.
- info: the water multiples inside an ACF window, at lags
  m 2 H sqrt(1 / vp_w^2 - p^2) (:func:`complybay.forward.water_multiple_lags`),
  and how far a water-depth error of ``DEPTH_ERROR_KM`` (10 m, assumed)
  shifts them, in seconds and in degrees at the peak of the processing
  passband.
- info: no water multiple inside the window, so the ACF fit does not depend
  on the water depth.

The multiples are the strongest arrivals of a seafloor ACF, so the water
depth should come from a bathymetric or acoustic survey of the station.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from complybay.forward import infragravity_cutoff, water_multiple_lags
from complybay.inference import ACFObservation, ComplianceObservation, Observation
from complybay.model import WaterLayer

# Assumed: the depth error used to express the water-multiple sensitivity.
DEPTH_ERROR_KM = 0.01


@dataclass(frozen=True)
class Note:
    level: Literal["warning", "info"]
    group: str
    message: str


def _passband_peak(acf: ACFObservation) -> float | None:
    """Frequency (Hz) of the peak of the combined filter power response."""
    if not acf.processing.filters:
        return None
    freqs = np.linspace(0.0, 0.5 / acf.dt, 4001)[1:]
    response = np.ones_like(freqs)
    for f in acf.processing.filters:
        response = response * f.power_response(freqs)
    return float(freqs[np.argmax(response)])


def check_observations(observations: Sequence[Observation], water: WaterLayer | None) -> list[Note]:
    notes = []
    for obs in observations:
        if isinstance(obs, ComplianceObservation) and water is not None:
            f_c = infragravity_cutoff(water.thickness * 1e3)
            above = obs.freqs > f_c
            if np.any(above):
                notes.append(
                    Note(
                        "warning",
                        obs.name,
                        f"{int(above.sum())} of {obs.freqs.size} frequencies above the "
                        f"infragravity limit {f_c:.4f} Hz for {water.thickness:g} km of water "
                        f"(up to {obs.freqs.max():.4f} Hz): the wave pressure is attenuated by "
                        "more than 268x at the seafloor there; drop them",
                    )
                )
        if isinstance(obs, ACFObservation):
            notes.extend(_check_acf(obs, water))
    return notes


def _check_acf(obs: ACFObservation, water: WaterLayer | None) -> list[Note]:
    notes = []
    lo, hi = obs.window
    max_lag = obs.lags[-1]
    if hi > max_lag:
        notes.append(
            Note(
                "warning",
                obs.name,
                f"window ends at {hi:g} s, beyond the data's last lag {max_lag:g} s",
            )
        )
    if lo <= 0 and obs.processing.normalize == "zero_lag":
        notes.append(
            Note(
                "warning",
                obs.name,
                "window includes the normalizing zero lag (1 in data and synthetics): it "
                "carries no information but counts in N; start the window above 0",
            )
        )
    if water is None:
        return notes
    first = water_multiple_lags(water, obs.ray_p, 1)[0]
    lags = water_multiple_lags(water, obs.ray_p, int(max_lag // first) + 1)
    in_window = (lags >= lo) & (lags <= min(hi, max_lag))
    inside = lags[in_window]
    order = np.arange(1, lags.size + 1)[in_window]  # m of each multiple, lag m * first
    if inside.size == 0:
        notes.append(
            Note(
                "info",
                obs.name,
                f"no water multiple inside the window (first at {lags[0]:.2f} s): the fit does "
                "not depend on the water depth",
            )
        )
        return notes
    q = lags[0] / (2 * water.thickness)
    shift = 2 * q * DEPTH_ERROR_KM * order
    peak = _passband_peak(obs)
    phase = (
        ""
        if peak is None
        else f" ({360 * shift[-1] * peak:.0f} deg at the passband peak {peak:.2f} Hz)"
    )
    notes.append(
        Note(
            "info",
            obs.name,
            f"water multiples inside the window at {', '.join(f'{t:.2f}' for t in inside)} s; "
            f"a {DEPTH_ERROR_KM * 1e3:g} m water-depth error shifts them by "
            f"{', '.join(f'{d:.3f}' for d in shift)} s{phase}",
        )
    )
    return notes


__all__ = ["DEPTH_ERROR_KM", "Note", "check_observations"]
