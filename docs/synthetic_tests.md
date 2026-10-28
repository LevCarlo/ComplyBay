# Synthetic tests

Inversions of known models that show what the data constrain and whether the
parameterization and the noise model are adequate. Each test is a script in
`tools/`, so its numbers can be regenerated; the numbers below come from the
runs named with each test.

## 1. What each data type constrains

`python tools/data_subset_test.py examples/synthetic runs/data_subsets`
(about 25 min with 4 processes per run) inverts the synthetic example
(`docs/workflow.md`; true model: basement 0.6 km, sediment Vs 0.15 -> 0.6 km/s,
Vp 1.75 km/s, 2 km of water) with every combination of compliance (C), phase
velocity (D), vertical ACF (Z) and pressure ACF (P). All data and the noise
realization are those of the example, and every case samples the four noise
scales of its groups as well. Run on commit 6144237 plus the script.
Figures: `tools/make_data_subsets_notebook.py` writes
`examples/data_subsets.ipynb`.

90 % interval width divided by the prior's (`*`: truth outside the 90 %
interval). `tp2`, `ts2`: vertical two-way P and S time through the sediment;
`vs_mean`: travel-time average sediment Vs, thickness / (ts2 / 2);
`vs_sqrt_mean`: (thickness average of sqrt(Vs))^2; `vs_mid`: Vs at mid-depth.
Derived quantities are computed from all kept samples.

| case | basement | vs_top | vs_bottom | vp | tp2 | ts2 | vs_mean | vs_sqrt_mean | vs_mid | max R-hat |
|---|---|---|---|---|---|---|---|---|---|---|
| C | 0.50 | 0.64 | 0.91 | 0.79 | 0.28 | 0.21 | 0.37 | 0.56 | 0.62 | 1.001 |
| D | 0.20 | 0.99 | 0.88 | 0.94 | 0.22 | 0.32 | 0.87 | 0.84 | 0.83 | 1.002 |
| Z | 0.04 | 1.03 | 0.84 | 0.15 | 0.01* | 0.51 | 1.06 | 1.02 | 1.00 | 1.010 |
| P | 0.04 | 1.01 | 0.74 | 0.14 | 0.01 | 0.47 | 0.94 | 0.88 | 0.87 | 1.001 |
| ZP | 0.03 | 1.03 | 0.83 | 0.11 | 0.01 | 0.53 | 1.08 | 1.03 | 1.01 | 1.018 |
| CD | 0.16 | 0.77 | 0.55 | 0.32 | 0.11 | 0.15 | 0.22 | 0.21 | 0.24 | 1.004 |
| CZP | (one chain stuck, see below) | | | | | | | | | 12.5 |
| DZP | 0.03 | 0.98 | 0.59 | 0.10 | 0.01 | 0.33 | 0.94 | 0.85 | 0.83 | 1.061 |
| CDZ | 0.04 | 0.49 | 0.30 | 0.14 | 0.01* | 0.13 | 0.19 | 0.07* | 0.09* | 1.054 |
| CDP | 0.03 | 0.72 | 0.38 | 0.13 | 0.01 | 0.13 | 0.20 | 0.08* | 0.09* | 1.002 |
| CDZP | 0.03 | 0.62 | 0.36 | 0.10 | 0.01* | 0.14 | 0.21 | 0.08* | 0.10* | 1.014 |

Correlation of the posterior samples (selected pairs):

| case | basement / vp | vs_top / vs_bottom | basement / vs_bottom |
|---|---|---|---|
| C | +0.97 | -0.73 | +0.90 |
| ZP | +0.97 | +0.55 | -0.27 |
| CD | +0.81 | -0.89 | +0.64 |
| CDZP | +0.96 | -0.95 | -0.07 |

What this shows:

- **The ACF fixes the sediment P two-way time.** Z or P alone narrows `tp2`
  to 1 % of the prior (0.686 s true; 90 % interval about 0.01 s wide), and with it basement depth
  and Vp to 4 % and 15 %, along a ridge basement / vp = tp2 / 2
  (correlation +0.97). The ACF says next to nothing about Vs.
- **Z and P carry about the same information** here; together they narrow
  basement and Vp only slightly further (0.03, 0.11). Both are worth fitting
  for robustness, not for resolution, as long as noise is independent between
  them.
- **Compliance constrains the S two-way time and the mean Vs** (`ts2` 0.20,
  `vs_mean` 0.37), not the individual parameters: basement and Vp trade off
  (+0.97), and so do the Vs end values (-0.73).
- **Dispersion (8-30 s) mostly sees sediment thickness** (basement 0.20,
  `tp2` 0.22), not sediment Vs (`vs_mean` 0.87; crust and mantle are fixed
  in this test, so "Vs" here is sediment Vs only). At these wavelengths
  (25-120 km) the 0.6 km sediment is a thin soft layer: how much crust it
  replaces matters, how soft it is hardly does. Change in phase velocity, in
  units of the data sigma (0.02 km/s), on the true model:

  | perturbation | 8 s | 12 s | 16 s | 20 s | 30 s |
  |---|---|---|---|---|---|
  | sediment Vs +20 % | 0.14 | 0.07 | 0.03 | 0.01 | 0.01 |
  | basement 0.1 km deeper | 2.08 | 0.59 | 0.32 | 0.21 | 0.11 |
  | sediment Vp +0.1 km/s | 0.32 | 0.00 | -0.01 | -0.01 | -0.01 |

  The sediment holds 15 % of the Vs kernel (sum of |dc/dVs| over layers) at
  8 s, 1.7 % at 16 s and 0.05 % at 30 s; the rest is crust and mantle, which
  test 2 samples. With compliance, dispersion still halves the `vs_mean`
  width (0.37 -> 0.21).
- **Joint inversion resolves what no single type does.** With all four groups,
  basement depth (0.59-0.62 km), Vp (1.71-1.80 km/s) and `tp2`
  (0.686-0.695 s) are tight. The sediment Vs profile is resolved only around
  a pivot, not at its ends: vs_top and vs_bottom are anti-correlated (-0.95),
  i.e. the gradient line rotates about a point 40-50 % down the sediment.
  CDZP, 90 % intervals of the linear profile:

  | depth in sediment | 0 | 0.12 km | 0.18 km | 0.24 km | 0.30 km | 0.36 km | 0.48 km | 0.60 km |
  |---|---|---|---|---|---|---|---|---|
  | Vs (km/s) | 0.06-0.37 | 0.21-0.38 | 0.28-0.39 | 0.34-0.40 | 0.38-0.44 | 0.40-0.51 | 0.42-0.66 | 0.43-0.81 |
  | true | 0.15 | 0.24 | 0.29 | 0.33 | 0.38 | 0.42 | 0.51 | 0.60 |

  The averages differ in how well they are fixed: the travel-time average
  (`vs_mean`, 0.29-0.40 km/s, +-16 %) and `ts2` (3.0-4.2 s) weight the slow,
  poorly resolved top heavily; their small widths relative to the prior
  (0.20, 0.14) mostly reflect a very wide prior (ts2 1.1-9.3 s). The
  best-fixed combination is close to the mean of sqrt(Vs) over the
  sediment, squared (`vs_sqrt_mean`, 0.37-0.42 km/s, +-6 %; 0.08 of the
  prior width against 0.21 for `vs_mean`), with `vs_mid` close behind. That Vs increases with depth is
  supported (P(vs_bottom > vs_top) = 0.97), its rate is not (0.1-1.2 km/s
  per km). So read the sediment through basement depth, Vp, `tp2` and Vs
  near mid-depth, plus the sign of the gradient; do not read the end values.
- **The pivot values miss the truth in this realization.** The true Vs at
  0.24 and 0.30 km (0.33, 0.38 km/s), and so `vs_sqrt_mean` and `vs_mid`, lie
  just below the CDZ, CDP and CDZP intervals. The
  cause is the noise draw, not the layering: at the highest frequencies,
  which see the shallow sediment, the observed compliance is 12-23 % below
  the noise-free value (stiffer, so faster), while the inversion layering
  changes compliance by chi2 = 0.0002 and phase velocity by chi2 = 0.0003.
  The tightest quantities are the most exposed to one noise draw; test 3
  (several noise seeds) checks the coverage.
- **Z puts `tp2` slightly late** (median 0.694 s against 0.686 s for P and
  the truth). Whether this is the noise draw or a systematic difference
  between the components will show with other noise seeds.
- **The truth just outside the `tp2` interval** (Z, CDZ, CDZP; 0.686 s
  against 0.6857-0.6953 for CDZP) is not a discretization bias: the ACF of
  the true model on the inversion layering differs from that on the fine
  layering used for the data by chi2 = 0.06 over 155 lags. All cases share one
  noise realization, so one 90 % miss in correlated cases is expected; repeating
  with other noise seeds (test 3) will tell.
- **The ACF makes the posterior multimodal.** In CZP one of four chains
  (start from the prior) stayed in a mode with basement 0.25 km and Vp 2.48
  km/s at its bound, `tp2` 0.20 s: a different reverberation lined up with
  the data, best log posterior -537 against -157 for the other chains, which
  agree with CDZP. Global jumps of one parameter did not get it out, because
  leaving needs basement and Vp to move together. Check the best log
  posterior per chain, and consider parallel tempering or more chains for
  real data.
