# Likelihood and posterior

`complybay.inference` turns data and the geophysical prior
(`docs/parameterization.md`) into the function a sampler explores:

```text
theta -> GeophysicalPrior.evaluate  (-inf + reason if outside; keeps the OBSModel)
      -> JointLikelihood.evaluate   (predict every observation, Gaussian misfit, noise scales)
      -> PosteriorTarget(theta) = log prior + log likelihood
```

## Observations

| class | data | settings |
|---|---|---|
| `ComplianceObservation(freqs, values, sigma)` | normalized compliance (1/Pa) | `wavenumber="exact"` |
| `DispersionObservation(periods, values, sigma)` | phase velocity (km/s) | `wave="rayleigh"`, `mode=0` |
| `ACFObservation(component, ray_p, processing, values, sigma, window)` | stacked ACF on lags 0..max_lag every `dt` | `dt`, `duration`, `snr`; only lags inside `window` (s) are fitted |

`sigma` may be a scalar or one value per sample. ACF observations with the
same `ray_p`, processing, `dt`, `duration` and `snr` (typically Z and P of one
stack) are predicted from one seafloor response.

## Likelihood and noise scales

For data group k with residuals r_i, standard deviations s_i and noise scale
h_k:

    log L = sum_k [ -1/2 sum_i (r_i / (h_k s_i))^2 - N_k ln h_k ]

Each observation names a sampled parameter holding `log10 h_k`
(`noise_parameter`), so a uniform prior on it is uniform in ln h, e.g.

```yaml
- {name: noise.compliance, initial: 0.0, prior: {bounds: [-0.5, 0.5]}}   # h in [0.32, 3.2]
```

The default design has four: compliance, dispersion, Z-ACF and P-ACF. The
`N_k ln h_k` term stops a scale from growing to swallow misfit; at fixed
residuals the best scale is h_k^2 = chi2_k / N_k. Without a noise parameter
the stated sigma is used as is.

Samples are treated as independent. ACF lags are correlated, so with the
stated sigma the ACF weight is too high by roughly the number of lags per
correlation length; the noise scale only corrects the overall level. A
correlated covariance is left for real-data work.

A solver exception or a non-finite prediction (for example a dispersion mode
with no root) gives log L = -inf with the reason.

## Posterior target

`PosteriorTarget(prior, likelihood)(theta)` is the log posterior. Vectors
outside the prior return -inf before any forward calculation.
`prior_only=True` returns the log prior alone: run the sampler on it first to
see the model space it explores.

Cost of one evaluation (30 solid layers, 12 frequencies, 8 periods, Z + P
ACF with `kim2023`): about 5.4 ms.

## Sampler

`run_metropolis(target, initial_thetas, MetropolisConfig(...), seed=..., executor=...)`
runs independent random-walk Metropolis chains (adapted from SeisForge
`seisforge.inv.samplers`, commit 019d568):

| option | meaning |
|---|---|
| `n_steps`, `burn_in`, `thin` | chain length; samples kept after burn-in, every `thin` steps |
| `proposal_sigma`, `bounds` | Gaussian step per parameter; proposals outside the box are rejected |
| `update` | `"one"`: perturb one random parameter per step (default); `"all"`: all at once; `"covariance"`: adaptive Metropolis, all at once from a learned covariance (below) |
| `adapt` | tune the sigmas during burn-in only (towards 0.44 / 0.234 acceptance), then freeze; with `"all"` only a common factor adapts |
| `covariance_start` | `"covariance"` only: step at which the covariance proposal takes over (default `burn_in // 2`) |
| `global_jump_interval` | every k-th step propose uniformly over the whole box (jumps between separated modes) |
| `temperature` | accept on d(log prior) + d(log likelihood) / T; T = 1 is the posterior. The hook for parallel tempering |

### Covariance proposals (`update="covariance"`)

Adaptive Metropolis (Haario, Saksman & Tamminen 2001) confined to burn-in:

1. Steps `0 .. covariance_start`: adaptive one-parameter updates (target
   acceptance 0.44), which find the scale of each parameter.
2. From `covariance_start // 2` to the end of burn-in, the chain's states feed
   a running mean and covariance C (Welford update).
3. From `covariance_start`: proposals theta' = theta + L z, L L^T =
   s^2 C + diag((1e-4 (upper - lower))^2), with s starting at 2.38 / sqrt(d)
   (Gelman, Roberts & Gilks 1996) and tuned by Robbins-Monro towards 0.234.
   The diagonal floor (assumed) keeps C positive definite.
4. At the end of burn-in L is frozen: the stored samples are an ordinary
   random-walk Metropolis chain, saved with `proposal_covariance` in
   `<stage>_mcmc.nc`.

Each chain learns its own C, so a chain that is still in a wrong mode at the
end of burn-in keeps a covariance of that mode; the global jumps (one
parameter, as for `"one"`) and R-hat across chains still matter. Burn-in must
be long enough for step 1 to reach the main mode and for step 2 to see the
ridge: a few thousand steps for about ten parameters.

On a 2-D Gaussian with correlation 0.99 (test suite) the effective sample size
of 2000 kept draws is about 24 with `"one"`, 120 with `"all"` and 2700 with
`"covariance"`.

`executor="process"` runs chains in parallel processes (the posterior target
is picklable); per-chain seeds are spawned from `seed`, so serial, thread and
process runs give identical samples. Start chains from
`sample_prior(prior, n_chains).theta` for diverse valid starting points.

First trial on the synthetic test model (9 parameters, noise-free data,
4 chains x 8000 steps, about 52 s wall in 4 processes): chains had not
converged. Their medians and best log posteriors differed, basement depth,
transition width and sediment Vs trade off along a narrow ridge, and all noise
scales went to their lower bound because noise-free data are best fitted with
h -> 0. Synthetic tests need noise added, longer chains, and a convergence
check (R-hat, effective sample size) before results are read.

The demo `examples/inversion_demo.ipynb` (noisy synthetic data from a finer
layering, hard basement, 4 model + 4 noise parameters, 4 chains x 30 000
steps, about 3.5 min): with `global_jump_interval=20` all chains reach the same
best log posterior and recover the true values within their 90 % intervals;
split R-hat is 1.09-1.12 for the model parameters and about 1.00 for the
noise scales. Without the global jumps one chain stayed in a wrong mode with
the basement near its 1.5 km bound. Sediment Vs at top and bottom trade off
(compliance constrains a depth average) and mix slowest.
