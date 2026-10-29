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
  (20 noise seeds) confirms it: their coverage is 18/20, as it should be.
- **Z puts `tp2` slightly late** (median 0.694 s against 0.686 s for P and
  the truth). Whether this is the noise draw or a systematic difference
  between the components will show with other noise seeds (test 3 finds a
  small late bias of tp2 from all data, +0.002 s).
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
  real data. Parallel tempering fixes this case (test 5).

## 2. Two sediment units over a sampled crust

`python tools/data_subset_test.py examples/synthetic_layered runs/layered
--cases CDZP CZP CD --segments sed_upper sed_lower` (about 15 min). True
model (`examples/synthetic_layered/truth.yaml`, assumed values): 2 km of
water; an upper unit 0-0.25 km (Vs 0.12 -> 0.35 km/s, Vp 1.6 km/s) over a
lower unit to 0.8 km (Vs 0.5 -> 0.9 km/s, Vp 2.0 km/s); crust to 7 km with
Vs 3.2 -> 3.8 km/s (Vp, rho from Brocher 2005), fixed mantle. 10 model
parameters (both interface depths, Vs top/bottom and Vp of each unit, crust
Vs top/bottom) and 4 noise scales; data and noise as in the example (seed 0).
CDZP was rerun with tempering (5 temperatures to 30; R-hat 1.019, ESS >= 351,
`runs/layered_pt`) and gave the same intervals within 0.02 km/s; the table
shows the tempered run for CDZP. 90 % intervals (`*`: truth outside):

| quantity | truth | CDZP | CZP | CD |
|---|---|---|---|---|
| sed_boundary depth (km) | 0.25 | 0.219-0.258 | 0.217-0.259 | 0.244-0.585 |
| basement depth (km) | 0.80 | 0.769-0.824 | 0.765-0.822 | 0.589-1.370 |
| upper Vp | 1.60 | 1.543-1.636 | 1.536-1.633 | 1.514-2.122 |
| lower Vp | 2.00 | 1.894-2.048 | 1.883-2.040 | 1.714-3.371 |
| upper vs_mean | 0.215 | 0.175-0.215 | 0.173-0.312 | 0.212-0.439 |
| upper vs_sqrt_mean | 0.230 | 0.186-0.247 | 0.184-0.318 | 0.226-0.461 |
| lower vs_top | 0.50 | 0.532-1.161* | 0.445-1.113 | 0.582-1.457* |
| lower vs_sqrt_mean | 0.695 | 0.692-0.852 | 0.686-0.848 | 0.660-1.492 |
| crust Vs top | 3.20 | 3.093-3.350 | 3.060-3.329 | 2.866-3.565 |
| crust Vs bottom | 3.80 | 3.641-3.919 | 3.666-3.966 | 3.474-4.162 |

R-hat (no tempering): CDZP 1.19, CZP 1.90, CD 1.08 (`upper.vs_top`); CZP
had one chain in a mode with upper Vs top 0.31 km/s whose log posterior
overlapped the others' (not a lagging chain, a weakly separated mode).

What this shows:

- **The ACF resolves both sediment units.** The intra-sediment boundary
  (0.22-0.26 km), the basement (0.77-0.82 km) and both Vp are fixed by the
  ACF, as tp2 was in test 1; without it (CD) neither depth is.
- **Sediment Vs: the upper unit's average is resolved, the lower unit's
  only loosely.** Compliance fixes the upper unit's mean Vs to about +-10 % once
  the ACF has fixed its thickness; the lower unit's mean Vs is known to about
  +-10 % on a 0.69-0.85 km/s range, and its gradient not at all (Vs top and
  bottom trade off; the truth of `lower.vs_top` lies just outside, as the Vs
  averages of test 1 did with the same noise seed). Each unit's Vs averages
  are the quantities to report, as in test 1.
- **Crust Vs is constrained by dispersion and by the ACF together.** Change
  in chi2 for crust Vs top +0.1 km/s at the truth: dispersion 3.7, ACF Z 5.0
  and P 3.6, compliance 0.1. The ACF part is crust Vp (Moho and crustal
  reverberations near 2 s lie inside the 0.3-8 s window) carried to Vs by
  the Brocher relation, so it holds only as far as that relation does. With
  dispersion and compliance alone (CD) the crust is barely constrained,
  because the unresolved sediment absorbs the short-period misfit.
- **Ten model parameters need longer chains or tempering.** Without
  tempering the soft upper unit mixed slowly (ESS 29 for its top Vs). With
  the default ladder the coldest swap was accepted only 7 % of the time
  (below the 10 % guideline): for 14 parameters use 6-7 temperatures.

## 3. Coverage over noise realizations

`python tools/coverage_test.py examples/synthetic runs/coverage --seeds 20`
(about 75 min): the example's data regenerated with noise seeds 0-19
(`tools/make_synthetic_example.py --seed`; seed 0 is the shipped data) and
inverted with all data groups and the example's sampler. Count of seeds
whose 90 % interval holds the truth (expected 18 +- 1.3), and the mean offset
of the median from the truth in units of the 90 % half-width:

| quantity | inside | offset | max R-hat |
|---|---|---|---|
| basement depth | 20 / 20 | +0.07 | 1.006 |
| sed.vs_top | 20 / 20 | +0.24 | 1.126 |
| sed.vs_bottom | 20 / 20 | -0.20 | 1.110 |
| sed.vp | 20 / 20 | -0.05 | 1.007 |
| tp2 | 17 / 20 | +0.44 | - |
| ts2 | 20 / 20 | -0.33 | - |
| vs_mean | 20 / 20 | +0.29 | - |
| vs_sqrt_mean | 18 / 20 | +0.22 | - |
| vs_mid | 18 / 20 | +0.10 | - |

- **The intervals are calibrated or conservative.** Every quantity's count
  lies within the binomial spread of 18 or above it. The wide, trading-off
  parameters (Vs end values, ts2, vs_mean) cover in every realization: their
  intervals are partly set by the prior bounds and the trade-off. The
  tightest Vs summaries (`vs_sqrt_mean`, `vs_mid`) cover as they should, so
  their misses in test 1 were the noise draw. One run of 20 had R-hat 1.13
  (sed.vs_top).
- **tp2 comes out slightly late.** The median exceeds the truth by 0.0020 s
  on average (standard error 0.0007 s over the seeds), 0.3 % of tp2 or 0.44
  of its half-width. Fitting the noise-free ACF of the fine layering with the
  inversion layering accounts for 0.0006 s of it (best basement depth
  0.6005 km instead of 0.6, chi2 0.09 over 310 lags); the rest, about two
  standard errors, is not explained. Against the tp2 interval width
  (+-0.0045 s) it is small, but the tp2 coverage (17/20) is the lowest, so
  treat tp2 intervals as slightly optimistic. Basement depth and Vp
  themselves show no bias.

## 4. Correlated ACF errors

`python tools/acf_correlation_test.py examples/synthetic runs/acf_correlation`
(about 15 min with 4 processes per run) fits `examples/synthetic/obs_events.yaml`:
compliance and dispersion of the example, and Z and P ACFs that are linear
stacks of 60 synthetic events of the true model with random 3 s source
wavelets and no additive noise (`tools/make_synthetic_example.py --events 60`),
so the stack errors come from the sources only. Three settings of
`Observations.acf.error_correlation` (`docs/inference.md`):

- `independent`: lags and components independent (the old likelihood),
- `processing`: the additive-noise theory, per component,
- `events`: estimated from the 60 event ACFs, Z and P jointly (shrinkage
  weight 0.075; 43 of 310 eigenvectors kept).

Run on commit 1ef243a plus the correlated-error changes. 90 % intervals
(`*`: truth outside); h: posterior median ACF noise scales (Z, P).

| case | basement | vp | tp2 | ts2 | vs_sqrt_mean | vs_mid | h (Z, P) | max R-hat |
|---|---|---|---|---|---|---|---|---|
| CDZP independent | 0.598-0.601 | 1.739-1.749* | 0.6868-0.6881* | 3.68-3.72 | 0.367-0.393* | 0.379-0.414* | 1.23, 1.39 | 1.377 |
| CDZP processing | 0.593-0.604 | 1.726-1.755 | 0.6862-0.6888* | 3.68-3.80 | 0.362-0.400 | 0.374-0.424 | 1.81, 1.43 | 1.049 |
| CDZP events | 0.594-0.601 | 1.730-1.749* | 0.6860-0.6878* | 3.63-3.71 | 0.364-0.389* | 0.376-0.408* | 2.24, 1.70 | 1.039 |
| ZP independent | 0.598-0.601 | 1.739-1.750* | 0.6868-0.6882* | 3.67-3.72 | 0.355-0.394 | 0.365-0.416 | 1.23, 1.36 | 3.050 |
| ZP events | 0.593-0.601 | 1.728-1.748* | 0.6859-0.6882* | 3.64-3.79 | 0.337-0.451 | 0.340-0.489 | 2.32, 1.71 | 1.403 |
| truth | 0.600 | 1.750 | 0.6857 | 3.70 | 0.363 | 0.375 | | |

What this shows:

- **Independent lags overstate what the ACF fixes.** With correlated errors
  the intervals of the quantities the ACF controls widen: CDZP basement depth
  by 1.8x (events) to 3x (processing), Vp 1.9x to 2.9x, `tp2` 1.4x to 2x;
  with the ACFs alone (ZP) basement and Vp by 2x and `vs_sqrt_mean` by 3x.
  The truth falls outside 6 of the 9 quantities' intervals with independent
  lags (CDZP), 2 with `processing` and 4 with `events`, the remaining misses
  being narrow (Vp 1.749 against 1.750, `vs_sqrt_mean` 0.364 against 0.363).
- **The stack does not equal the forward ACF of the true model.** At the truth
  with h = 1 the ACFs give chi2/N = 1.6 (Z) and 2.2 (P) with lags taken as
  independent, and chi2/k = 4.8 in the correlated block (k = 43): the mean
  over events of the windowed, whitened event ACF differs from the
  spectral-ratio prediction by about twice the stack sigma (std / sqrt(60)),
  which only describes the scatter between events. So h rises above 1 (1.4-2.3)
  in every case, and `tp2` is missed in all of them by 0.0002-0.0011 s, a bias
  that more events would not remove. On real data structure misfit adds to
  this: keep the ACF noise scales free and read h as the size of the
  unmodelled part.
- **Which correlation to use.** `events` keeps the Z-P cross-correlation and
  the source-generated correlation that `processing` lacks, and is preferred
  when per-event ACFs exist; `processing` assumes more independent pieces of
  information than the source-generated errors have, but here gave the wider
  intervals, because its h is free to absorb the difference. Either is better
  than independent lags.
- **Noise-free stacks make the ACF fix ts2.** Unlike test 1, the ACFs alone
  narrow `ts2` (3.64-3.79 s): with no additive noise the stack sigma is small
  enough for the converted phases to count. This is specific to the
  noise-free events, not a property of real stacks.
- **Convergence.** ZP independent (R-hat 3.05) and ZP events (1.40) did not
  converge in 30000 steps: without compliance the sediment Vs is weakly fixed
  and chains stay in separate modes, as in test 1's CZP case.

## 5. Parallel tempering on the stuck case

The CZP case of test 1 rerun with `Sampler.tempering` (same seed, so the same
starting points, including the one that got stuck), on the code of the
tempering change. Log posterior per chain: median (best).

| run | chain 0 | chain 1 | chain 2 | chain 3 | max R-hat | ESS basement / vp | swap acceptance | sampling time |
|---|---|---|---|---|---|---|---|---|
| no tempering (test 1) | -159.8 (-156.7) | -540.9 (-537.0) | -159.8 (-156.9) | -160.0 (-156.8) | 12.5 | - | - | 147 s |
| 5 temperatures, T_max 30 | -159.8 (-156.8) | -159.9 (-156.8) | -159.9 (-156.5) | -160.0 (-156.8) | 1.001 | 3346 / 3351 | 0.24-0.36 | 724 s |
| 8 temperatures, T_max 100 | -159.9 (-156.7) | -159.9 (-156.5) | -159.8 (-156.7) | -160.0 (-156.9) | 1.002 | 3560 / 3685 | 0.37-0.62 | 1065 s |

90 % intervals with tempering (both ladders agree within 0.001): basement
0.591-0.622 km, Vp 1.71-1.80 km/s, vs_top 0.06-0.33, vs_bottom 0.46-0.81
km/s; the truth (0.60, 1.75, 0.15, 0.60) lies inside all of them.

- **The stuck chain is pulled out.** All four chains now reach the same log
  posterior; `lagging_chains` flags chain 1 of the untempered run and none of
  the tempered ones.
- **5 temperatures up to 30 suffice here**, the workflow default. The larger
  ladder accepts more swaps but gives the same posterior at 1.5x the cost.
  Swap acceptance well above 0.1 on every pair means the ladder could be
  coarser; acceptance near 0 on a pair means a temperature is missing there.
- **Cost.** A ladder of n costs about n times a plain run (here 4.9x for
  n = 5 and 7.2x for n = 8). For real data, run untempered first; turn tempering on when
  `lagging_chains` or R-hat point to separate modes, or from the start when the
  ACF is fitted without compliance, the case that got stuck here.

## 6. Attenuating sediment

`python tools/q_synthetic_test.py` (about 17 min) makes data from the true
model of the example with sediment Qs = 20 and Qp = 100 (velocities at the
1 Hz reference; `docs/forward.md`, attenuation) and the example's noise
realization, and inverts them elastically and with Q fixed at the truth. All
data groups, the example's sampler. The noise realization is the one of test
1, which already puts the sediment Vs averages about 6 % above the truth, so
each run is compared with test 1's CDZP case (elastic data, elastic
inversion, same noise); ratio of medians in brackets.

| quantity | elastic data, elastic (test 1) | Q data, elastic | Q data, Q fixed |
|---|---|---|---|
| basement (km) | 0.591-0.622 | 0.600-0.631 (1.014) | 0.591-0.621 (0.999) |
| Vp (km/s) | 1.71-1.80 | 1.74-1.83 (1.014) | 1.71-1.80 (0.999) |
| tp2 (s) | 0.686-0.695 | 0.686-0.696 (1.000) | 0.686-0.696 (1.000) |
| vs_sqrt_mean (km/s) | 0.374-0.418 | 0.350-0.390 (0.943) | 0.374-0.413 (0.995) |
| vs_mid (km/s) | 0.382-0.440 | 0.356-0.404 (0.933) | 0.381-0.432 (0.992) |

- **With Q in the model the result is that of the elastic test**: the
  attenuating forward solvers recover what the elastic ones did, within 1 %.
- **An elastic inversion returns the sediment Vs of the compliance band.**
  Its Vs averages are 6-7 % below those at 1 Hz, as expected for Qs = 20:
  the constant-Q phase velocity at 0.01-0.05 Hz is 0.93-0.95 of that at
  1 Hz. Compliance carries the Vs information (test 1), so the elastic
  model's Vs is the Vs at compliance frequencies, not at the ACF's.
  Basement depth and Vp move by 1.4 %, the ACF's share of the dispersion.
- **What to do with real data.** If sediment Q is unknown, say so when
  reporting Vs: an elastic result is Vs at about 0.01 Hz. To compare with Vs
  at 1 Hz or higher (logs, refraction, geotechnical relations), either fix Q
  from an independent estimate or apply the correction above. Whether the
  data constrain Q themselves has not been tested; Qs barely changes the
  ACF (`docs/forward.md`), so a sampled Qs would mostly trade off with Vs
  through the dispersion between bands and needs an informative prior.
- **Cost**: 772 s against 207 s for the elastic run (3.7x).
