"""Figure likelihood_correlation.png: correlated ACF errors and the reduced-rank likelihood.

Monte Carlo of the processing: 6000 white-noise traces (70 s at 0.05 s, the
Kim et al. 2023 window) through the full kim2023 processing
(``acf_from_trace``); the first 3000 give the sample correlation over the
0.3-8 s window (155 lags), the other 3000 are test residuals. Also writes
the table of effective numbers of data, docs/generated/likelihood_numbers.md.

(a) Rows of the correlation at lags 1 s and 4 s: processing theory
(``processing_error_correlation``, lines) against the Monte Carlo (dots).
(b) Eigenvalues of the correlation, as a fraction of the trace: theory,
Monte Carlo, and the event estimate shipped with examples/synthetic
(``acf_errors.nc``: 60 synthetic events with random source wavelets, Z block);
dots: the cut at 99 % of the variance.
(c) Noise scale that maximizes the likelihood of each test residual, sqrt of
chi2 / (degrees of freedom), with the theory correlation: lags independent
(chi2 / N), reduced rank (|y|^2 / k, the package), full rank with a 1e-3
nugget (chi2 / N). Known answer: h = 1 (the residuals have unit variance).

Run: python docs/scripts/likelihood_correlation.py
"""

from pathlib import Path

import _style
import numpy as np
from _style import GENERATED_DIR, plt

from complybay.acf import acf_from_trace, preset, processing_error_correlation
from complybay.inference import ErrorCorrelation
from complybay.io import read_acf_errors

DT, N_SAMPLES, N_TRACES = 0.05, 1400, 6000
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "synthetic"
kim = preset("kim2023")
rng = np.random.default_rng(1)
acfs = np.array(
    [
        acf_from_trace(rng.standard_normal(N_SAMPLES), DT, kim, taper_fraction=0.2)[1]
        for _ in range(N_TRACES)
    ]
)
lags = np.arange(acfs.shape[1]) * DT
fit = (lags >= 0.3) & (lags <= 8.0)
cal, test = acfs[: N_TRACES // 2, fit], acfs[N_TRACES // 2 :, fit]
n = int(fit.sum())

simulated = np.corrcoef(cal.T)
theory = processing_error_correlation(kim, lags[fit], DT, expected=acfs[: N_TRACES // 2].mean(0))
# 3000 traces: standard error of a correlation ~0.018; the max over ~12000 pairs ~4.5 of them.
assert np.max(np.abs(simulated - theory)) < 0.1
assert np.sqrt(np.mean((simulated - theory) ** 2)) < 0.03

events = read_acf_errors(EXAMPLE / "acf_errors.nc")
p = events.window_lags.size
event_z = events.correlation[:p, :p]


def n_eff(r):
    return np.trace(r) ** 2 / np.sum(r**2)


block = ErrorCorrelation(("acf",), theory)
z = (test - cal.mean(axis=0)) / cal.std(axis=0)
h_independent = np.sqrt(np.sum(z**2, axis=1) / n)
h_reduced = np.sqrt(np.sum((z @ block.basis) ** 2, axis=1) / block.k)
nugget = 1e-3  # assumed size of a diagonal regularization, for illustration
r_nugget = (theory + nugget * np.eye(n)) / (1 + nugget)
h_full = np.sqrt(np.einsum("ij,jk,ik->i", z, np.linalg.inv(r_nugget), z) / n)
chi2_full_no_nugget = np.einsum("ij,jk,ik->i", z, np.linalg.pinv(theory, rcond=1e-15), z) / n
# Reduced rank: chi2 / k ~ 1 (residuals of the real processing, theory correlation).
assert abs(np.mean(h_reduced**2) - 1) < 0.05
assert np.median(h_full) < 0.7 and np.median(chi2_full_no_nugget) > 100

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(10.0, 3.2))
for lag, color in ((1.0, _style.PACKAGE), (4.0, _style.SECOND)):
    i = int(np.argmin(np.abs(lags[fit] - lag)))
    ax_a.plot(lags[fit], theory[i], color=color, lw=1.2, label=f"theory, row {lag:g} s")
    ax_a.plot(lags[fit][::2], simulated[i, ::2], "o", ms=2, color=color, alpha=0.7)
ax_a.plot([], [], "o", ms=2, color="0.3", label="Monte Carlo")
ax_a.set_xlabel("lag (s)")
ax_a.set_ylabel("error correlation")
ax_a.legend(loc="upper right")
_style.panel_label(ax_a, "(a)")

numbers = {}
for name, r, color in (
    ("theory (white noise)", theory, _style.TRUTH),
    ("Monte Carlo (white noise)", simulated, _style.PACKAGE),
    ("60 events, Z (acf_errors.nc)", event_z, _style.SECOND),
):
    values = np.clip(np.linalg.eigvalsh(r)[::-1], 1e-12, None) / np.trace(r)
    k = ErrorCorrelation(("x",), r).k
    style = {"ls": "--", "zorder": 3} if r is theory else {}
    ax_b.semilogy(np.arange(1, values.size + 1), values, color=color, lw=1.2, label=name, **style)
    ax_b.plot(k, values[k - 1], "o", color=color, ms=4)
    numbers[name] = (n_eff(r), k)
ax_b.set_ylim(1e-8, 1)
ax_b.set_xlabel("eigenvalue index")
ax_b.set_ylabel(r"$\lambda_i / \mathrm{tr}\,R$")
ax_b.legend(loc="upper right")
_style.panel_label(ax_b, "(b)")

bins = np.linspace(0.2, 2.0, 46)
for values, label, color in (
    (h_independent, "lags independent", _style.APPROX),
    (h_reduced, f"reduced rank (k = {block.k})", _style.PACKAGE),
    (h_full, "full rank + 1e-3 nugget", _style.HIGHLIGHT),
):
    ax_c.hist(values, bins=bins, histtype="step", lw=1.3, color=color, label=label)
ax_c.axvline(1.0, color=_style.TRUTH, lw=0.8)
ax_c.set_xlabel(r"best noise scale $\hat h$")
ax_c.set_ylabel("residuals")
ax_c.legend(loc="upper right")
_style.panel_label(ax_c, "(c)")
fig.tight_layout()
_style.save(fig, __file__)

# Effective number of data and retained eigenvectors.
joint_k = ErrorCorrelation(("z", "p"), events.correlation).k
cross = np.mean(np.diag(events.correlation[:p, p:]))
lines = [
    "| correlation (0.3-8 s, 155 lags per component) | $N_\\mathrm{eff}$ | $k$ at 99 % |",
    "|---|---|---|",
]
for name, (value, k) in numbers.items():
    lines.append(f"| {name} | {value:.1f} | {k} |")
lines.append(
    f"| 60 events, Z and P jointly (310 samples) | {n_eff(events.correlation):.1f} | {joint_k} |"
)
(GENERATED_DIR / "likelihood_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(
    f"shrinkage weight of acf_errors.nc: {events.shrinkage:.3f}; mean Z-P correlation at equal "
    f"lag {cross:.2f}"
)
print(
    f"mean chi2/dof: independent {np.mean(h_independent**2):.3f} "
    f"(sd {np.std(h_independent**2):.2f},"
    f" chi2_N/N sd {np.sqrt(2 / n):.2f}); reduced {np.mean(h_reduced**2):.3f} "
    f"(sd {np.std(h_reduced**2):.2f}, chi2_k/k sd {np.sqrt(2 / block.k):.2f}); "
    f"full+nugget median h {np.median(h_full):.2f}; full rank no nugget median chi2/N "
    f"{np.median(chi2_full_no_nugget):.3g}; sqrt(k/N) = {np.sqrt(block.k / n):.2f}"
)
print(
    f"max |theory - MC| {np.max(np.abs(simulated - theory)):.3f}, "
    f"rms {np.sqrt(np.mean((simulated - theory) ** 2)):.3f}"
)
