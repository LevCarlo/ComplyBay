"""Figure acf_reading.png: where the peaks of a seafloor ACF come from.

The shared docs model (2 km of water over 0.6 km of sediment with Vp 1.8 km/s,
crust and mantle) at ray parameter p = 0.06 s/km. (a) Vertical and (b)
pressure ACFs with both presets. Vertical lines mark the predicted two-way
vertical times

    tau = 2 sum_i h_i q_i,   q_i = sqrt(1 / v_i^2 - p^2)

of the sediment P reverberation (tau_s), the water multiples (m tau_w, from
``water_multiple_lags``), their sums, the crustal P reverberation from the
seafloor to the Moho (tau_M) and the sediment S reverberation (tau_S). The
formula is written to docs/generated/acf_lags.md.

Checks: ``water_multiple_lags`` equals 2 H q_w; the largest peak after 1 s lies
within two samples (0.1 s) of tau_w or tau_w + tau_s on every trace; the
water multiple has opposite signs on Z and P; the tauzin2019 Z-ACF has a
negative peak within 0.05 s of tau_s and the pressure ACF a positive one.

Run: python docs/scripts/acf_reading.py
"""

from __future__ import annotations

import _style
import numpy as np
from _models import sediment_over_crust
from _style import GENERATED_DIR, plt

from complybay.acf import preset
from complybay.forward.acf import synthetic_acfs, water_multiple_lags

RAY_P = 0.06  # s/km
model = sediment_over_crust()
solid, water = model.solid, model.water


def two_way(h, v):
    return 2 * np.sum(np.asarray(h) * np.sqrt(1 / np.asarray(v) ** 2 - RAY_P**2))


tau_w = two_way(water.thickness, water.vp)
assert np.isclose(water_multiple_lags(water, RAY_P, 1)[0], tau_w, rtol=1e-14)
tau_s = two_way(solid.thickness[0], solid.vp[0])
tau_s_shear = two_way(solid.thickness[0], solid.vs[0])
tau_moho = two_way(solid.thickness[:3], solid.vp[:3])  # seafloor to Moho (6.6 km)
marks = {
    r"$\tau_s$": tau_s,
    r"$\tau_M$": tau_moho,
    r"$\tau_w$": tau_w,
    r"$\tau_w{+}\tau_s$": tau_w + tau_s,
    r"$\tau_w{+}2\tau_s$": tau_w + 2 * tau_s,
    r"$\tau_S$": tau_s_shear,
    r"$2\tau_w$": 2 * tau_w,
    r"$2\tau_w{+}\tau_s$": 2 * tau_w + tau_s,
    r"$2\tau_w{+}2\tau_s$": 2 * tau_w + 2 * tau_s,
}

GENERATED_DIR.mkdir(exist_ok=True)
(GENERATED_DIR / "acf_lags.md").write_text(
    "$$\n\\boxed{\\;\\tau = 2\\sum_i h_i\\, q_i,"
    "\\qquad q_i = \\sqrt{\\frac{1}{v_i^2} - p^2}\\;}\n$$\n\n"
    "| arrival | layers, velocity | $\\tau$ (s), docs model, $p$ = 0.06 s/km |\n|---|---|---|\n"
    f"| sediment P, $\\tau_s$ | sediment, $v_p$ | {tau_s:.2f} |\n"
    f"| water multiple, $\\tau_w$ | water, $v_{{p,w}}$ | {tau_w:.2f} |\n"
    f"| crustal P, $\\tau_M$ | seafloor to Moho, $v_p$ | {tau_moho:.2f} |\n"
    f"| sediment S, $\\tau_S$ | sediment, $v_s$ | {tau_s_shear:.2f} |\n",
    encoding="utf-8",
)

fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
results = {}
for name, color in zip(("tauzin2019", "kim2023"), _style.SERIES, strict=False):
    lags, acfs = synthetic_acfs(model, RAY_P, preset(name))
    for ax, component in zip(axes, ("vertical", "pressure"), strict=True):
        acf = acfs[component]
        results[name, component] = acf
        ax.plot(lags, acf, color=color, lw=1.1, label=name)
        late = lags > 1.0
        peak = lags[late][np.argmax(np.abs(acf[late]))]
        assert min(abs(peak - tau_w), abs(peak - tau_w - tau_s)) < 0.1, (name, component, peak)


def value_near(acf, lag, half_width=0.1):
    near = np.abs(lags - lag) <= half_width
    return acf[near][np.argmax(np.abs(acf[near]))]


for name in ("tauzin2019", "kim2023"):
    z, p = (value_near(results[name, c], tau_w) for c in ("vertical", "pressure"))
    assert np.sign(z) == -np.sign(p), (name, z, p)
z = results["tauzin2019", "vertical"]
near = np.abs(lags - tau_s) <= 0.05
assert z[near].min() < -0.2, z[near]
p_acf = results["tauzin2019", "pressure"]
assert p_acf[near].max() > 0.2, p_acf[near]  # opposite sign on P in this model

for ax, title, letter in zip(axes, ("vertical (Z)", "pressure (P)"), ("(a)", "(b)"), strict=True):
    for i, (label, lag) in enumerate(marks.items()):
        ax.axvline(lag, color=_style.APPROX, lw=0.7, ls=":")
        if ax is axes[0]:
            ax.text(lag, 1.08 + 0.13 * (i % 2), label, ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color=_style.TRUTH, lw=0.5)
    ax.set_ylim(-1.1, 1.05)
    ax.set_ylabel(f"ACF, {title}")
    _style.panel_label(ax, letter)
axes[0].legend(loc="lower right", ncols=2)
axes[1].set_xlim(0, 8)
axes[1].set_xlabel("lag (s)")
fig.tight_layout()
_style.save(fig, __file__)
print(", ".join(f"{k} = {v:.2f}" for k, v in marks.items()))
