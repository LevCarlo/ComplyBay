"""Write examples/data_subsets.ipynb, the figures of tools/data_subset_test.py.

Notebooks are not tracked; regenerate and run it from the repository root:

    python tools/make_data_subsets_notebook.py
    jupyter nbconvert --to notebook --execute --inplace examples/data_subsets.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# What each data type constrains

Results of `tools/data_subset_test.py`: the synthetic example inverted with every
combination of compliance (C), phase velocity (D), vertical ACF (Z) and pressure ACF (P).
Run first:

```bash
python tools/data_subset_test.py examples/synthetic runs/data_subsets
```

Discussion in `docs/synthetic_tests.md`."""),
    code("""from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

RUNS = Path("../runs/data_subsets")
s = xr.open_dataset(RUNS / "summary.nc", engine="h5netcdf")
cases = list(s.case.values)
LABELS = {"basement.depth": "basement depth (km)", "sed.vs_top": "sediment Vs top (km/s)",
          "sed.vs_bottom": "sediment Vs bottom (km/s)", "sed.vp": "sediment Vp (km/s)",
          "tp2.sediment": "P two-way time (s)", "ts2.sediment": "S two-way time (s)",
          "vs_mean.sediment": "travel-time mean Vs (km/s)",
    "vs_sqrt_mean.sediment": "sqrt-mean Vs (km/s)",
    "vs_mid.sediment": "mid-depth Vs (km/s)"}
INK, MUTED, MARK, TRUTH = "#1f2328", "#8b929a", "#2a6fdb", "#c2410c"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK,
                     "axes.labelcolor": INK, "font.size": 9})"""),
    md("""## Stuck chains

Best log posterior per chain. A chain far below the others sits in a wrong mode
(CZP: one chain at basement 0.25 km, Vp at its bound)."""),
    code("""rows = []
for case in cases[1:]:
    with xr.open_dataset(RUNS / case / "posterior_mcmc.nc", engine="h5netcdf") as m:
        best = (m.log_prior + m.log_likelihood).max("draw").values
    rows.append((case, best))
    flag = "  <- chain %d stuck" % np.argmin(best) if best.max() - best.min() > 20 else ""
    print(f"{case:5s}", " ".join(f"{b:8.1f}" for b in best), flag)"""),
    md("""## 90 % intervals per case

Blue: posterior 5-95 % with median; grey band: prior 5-95 %; orange line: true value.
Each panel is one quantity, so each has its own axis."""),
    code("""names = list(s.quantity.values)
fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharey=True)
y = np.arange(len(cases))[::-1]
for ax, name in zip(axes.flat, names):
    q = s.sel(quantity=name)
    ax.axvspan(float(q.q05[0]), float(q.q95[0]), color=MUTED, alpha=0.15, lw=0)
    ax.hlines(y[1:], q.q05[1:], q.q95[1:], color=MARK, lw=2)
    ax.plot(q["median"][1:], y[1:], "o", color=MARK, ms=4)
    ax.axvline(float(q.truth), color=TRUTH, lw=1.2)
    ax.set_title(LABELS[name], fontsize=9, color=INK)
    ax.grid(axis="x", color=MUTED, alpha=0.2, lw=0.5)
axes.flat[-1].axis("off")
axes[0, 0].set_yticks(y, cases); axes[1, 0].set_yticks(y, cases)
fig.tight_layout()"""),
    md("""Zoom on the well-constrained quantities (prior band off-scale)."""),
    code("""fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, name in zip(axes, ["basement.depth", "sed.vp", "tp2.sediment", "vs_sqrt_mean.sediment"]):
    q = s.sel(quantity=name)
    ax.hlines(y[1:], q.q05[1:], q.q95[1:], color=MARK, lw=2)
    ax.plot(q["median"][1:], y[1:], "o", color=MARK, ms=4)
    ax.axvline(float(q.truth), color=TRUTH, lw=1.2)
    ax.set_title(LABELS[name], fontsize=9)
    keep = [c for c in cases[1:] if c not in ("C", "D", "CD", "CZP")]
    lo = float(q.q05.sel(case=keep).min()); hi = float(q.q95.sel(case=keep).max())
    ax.set_xlim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.grid(axis="x", color=MUTED, alpha=0.2, lw=0.5)
axes[0].set_yticks(y[1:], cases[1:])
fig.tight_layout()"""),
    md("""## Relative width (posterior 90 % / prior 90 %)"""),
    code("""rel = s.relative_width.values[1:]
fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(rel, cmap="Blues_r", vmin=0, vmax=1.1, aspect="auto")
ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=35, ha="right")
ax.set_yticks(range(len(cases) - 1), cases[1:])
for i in range(rel.shape[0]):
    for j in range(rel.shape[1]):
        ax.text(j, i, f"{rel[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if rel[i, j] < 0.45 else INK)
fig.colorbar(im, ax=ax, label="relative 90 % width")
fig.tight_layout()"""),
    md("""## Correlation of the model parameters

Diverging scale, grey at zero."""),
    code("""from matplotlib.colors import LinearSegmentedColormap
div = LinearSegmentedColormap.from_list("div", ["#2a6fdb", "#e5e7eb", "#c2410c"])
show = ["C", "D", "ZP", "CD", "DZP", "CDZP"]
params = [p.split(".")[-1] for p in s.parameter.values]
fig, axes = plt.subplots(1, len(show), figsize=(15, 3.2), layout="constrained")
for k, (ax, case) in enumerate(zip(axes, show)):
    c = np.ma.masked_where(np.eye(4, dtype=bool), s.correlation.sel(case=case).values)
    im = ax.imshow(c, cmap=div, vmin=-1, vmax=1)
    ax.set_xticks(range(4), params, rotation=60, fontsize=7)
    ax.set_yticks(range(4), params if k == 0 else [], fontsize=7)
    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f"{c[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_title(case)
fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")"""),
]
nb.metadata["kernelspec"] = {
    "name": "complybay",
    "display_name": "Python (complybay)",
    "language": "python",
}
nbf.write(nb, "examples/data_subsets.ipynb")
