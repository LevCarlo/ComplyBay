"""Figure acf_whitening.png: what running-mean whitening does to short lags.

A single echo, H(f) = 1 + eps exp(2 pi i f tau0), puts a ripple
2 eps cos(2 pi f tau0) on |H|^2 and a peak at lag tau0 in the ACF. Whitening
divides by a running mean over W Hz, which keeps a fraction sinc(W tau0) of
the ripple, so to first order in eps the peak is multiplied by
1 - sinc(W tau0) (derived on docs/acf.md; written to
docs/generated/acf_sinc.md).

The script measures the factor with the package: the echo term
ACF(eps) - ACF(0) at lag tau0, whitened over unwhitened, for both presets
(their whitening and filters, no zero-lag taper, no normalization). (a) The
factor against tau0: package (dots), 1 - sinc(W tau) (lines) and the exact
discrete running mean of 2N + 1 samples (dashed). (b) The echo term of a
0.7 s echo with and without kim2023 whitening.

Run: python docs/scripts/acf_whitening.py
"""

from __future__ import annotations

from dataclasses import replace

import _style
import numpy as np
from _style import GENERATED_DIR, plt
from scipy.fft import next_fast_len, rfftfreq

from complybay.acf import acf_from_spectrum, preset

DT, DURATION, EPS = 0.05, 200.0, 0.01
N_FFT = next_fast_len(int(round(DURATION / DT)), real=True)
FREQS = rfftfreq(N_FFT, DT)
DF = FREQS[1]


def echo_term(processing, tau0, eps=EPS):
    """ACF(eps) - ACF(0) of H = 1 + eps exp(2 pi i f tau0) (lags, values)."""
    plain = replace(processing, zero_lag_taper=0.0, normalize="none")
    lags, with_echo = acf_from_spectrum(
        1 + eps * np.exp(2j * np.pi * FREQS * tau0), DT, N_FFT, plain
    )
    _, without = acf_from_spectrum(np.ones(FREQS.size, complex), DT, N_FFT, plain)
    return lags, with_echo - without


def measured_factor(processing, tau0):
    lags, whitened = echo_term(processing, tau0)
    _, raw = echo_term(replace(processing, whitening=None), tau0)
    i = int(round(tau0 / DT))
    return whitened[i] / raw[i]


def continuous(width, tau):
    return 1 - np.sinc(width * tau)  # np.sinc(x) = sin(pi x) / (pi x)


def discrete(width, tau):
    """1 - (mean of cos(2 pi f tau) over 2N + 1 samples) / cos(2 pi f tau)."""
    n = 2 * int(round(width / (2 * DF))) + 1
    x = np.pi * DF * np.asarray(tau)
    return 1 - np.sin(n * x) / (n * np.sin(x))


taus = np.arange(0.2, 5.01, 0.1)
results = {}
for name in ("kim2023", "tauzin2019"):
    processing = preset(name)
    width = processing.whitening.width
    factor = np.array([measured_factor(processing, t) for t in taus])
    results[name] = (width, factor)
    # First order in eps: the measured factor is the discrete running-mean factor
    # (largest difference 2.3e-3, from O(eps) terms and the band edges, where the
    # running mean is truncated) and within 0.015 of the continuous form
    # 1 - sinc(W tau) for tau >= 0.2 s (the discrete window is 2N + 1 samples wide,
    # W + df instead of W).
    assert np.max(np.abs(factor - discrete(width, taus))) < 5e-3, name
    assert np.max(np.abs(factor - continuous(width, taus))) < 0.015, name
    print(
        f"{name}: W = {width} Hz, |package - discrete| <= "
        f"{np.max(np.abs(factor - discrete(width, taus))):.1e}, |package - (1 - sinc)| <= "
        f"{np.max(np.abs(factor - continuous(width, taus))):.1e}"
    )

GENERATED_DIR.mkdir(exist_ok=True)
w_kim, w_tauzin = results["kim2023"][0], results["tauzin2019"][0]
(GENERATED_DIR / "acf_sinc.md").write_text(
    "$$\n\\boxed{\\;\\mathrm{ACF}_\\mathrm{whitened}(\\tau) \\approx"
    " \\left[1 - \\operatorname{sinc}(W\\tau)\\right]\\mathrm{ACF}(\\tau)\\;}"
    "\\qquad \\operatorname{sinc} x = \\frac{\\sin \\pi x}{\\pi x}\n$$\n\n"
    "| lag $\\tau$ (s) | 0.3 | 0.5 | 0.7 | 1.0 | 1.5 | 2.0 | 3.0 |\n"
    "|---|---|---|---|---|---|---|---|\n"
    + "".join(
        f"| $W$ = {w:g} Hz | "
        + " | ".join(f"{continuous(w, t):.2f}" for t in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0))
        + " |\n"
        for w in (w_kim, w_tauzin)
    ),
    encoding="utf-8",
)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
fine = np.linspace(0.05, 5.0, 400)
for (name, (width, factor)), color in zip(results.items(), _style.SERIES, strict=False):
    ax_a.plot(
        fine,
        continuous(width, fine),
        color=color,
        lw=1.2,
        label=rf"$1 - \mathrm{{sinc}}(W\tau)$, $W$ = {width:g} Hz",
    )
    ax_a.plot(fine, discrete(width, fine), color=color, lw=0.8, ls="--")
    ax_a.plot(taus, factor, "o", ms=3, color=color, label=f"package, {name}")
ax_a.axhline(1, color=_style.TRUTH, lw=0.6)
ax_a.set_xlabel(r"echo lag $\tau$ (s)")
ax_a.set_ylabel("whitened / unwhitened echo")
ax_a.legend(loc="lower right")
_style.panel_label(ax_a, "(a)")

kim = preset("kim2023")
lags, whitened = echo_term(kim, 0.7)
_, raw = echo_term(replace(kim, whitening=None), 0.7)
scale = raw[int(round(0.7 / DT))]
ax_b.plot(lags, raw / scale, color=_style.APPROX, lw=1.2, label="no whitening")
ax_b.plot(lags, whitened / scale, color=_style.PACKAGE, lw=1.2, label="kim2023 whitening")
ax_b.axvline(0.7, color=_style.TRUTH, lw=0.6, ls=":")
ax_b.set_xlim(0, 3)
ax_b.set_xlabel("lag (s)")
ax_b.set_ylabel("echo term (scaled)")
ax_b.set_title(r"echo at $\tau_0$ = 0.7 s, kim2023 filters")
ax_b.legend(loc="upper right")
_style.panel_label(ax_b, "(b)")
fig.tight_layout()
_style.save(fig, __file__)
