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
