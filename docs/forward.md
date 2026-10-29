# Forward solvers

Three observables share one model, `complybay.model.OBSModel`: an optional
`WaterLayer` over a `LayeredModel` (n solid layers plus a half-space).
Units everywhere: km, km/s, g/cm^3; depth 0 is the seafloor.

```python
from complybay.model import LayeredModel, OBSModel, WaterLayer
from complybay.forward import compliance, phase_velocity, synthetic_acf
from complybay.acf import preset

solid = LayeredModel(
    thickness=[0.4, 1.5, 5.0, 53.0],
    vp=[1.8, 4.5, 6.5, 8.1, 7.9],
    vs=[0.3, 2.5, 3.7, 4.55, 4.3],
    rho=[1.8, 2.6, 2.9, 3.35, 3.35],
)
model = OBSModel(solid, WaterLayer(4.0))

eta = compliance(model, [0.005, 0.01, 0.02])                 # 1/Pa
c = phase_velocity(model, [8.0, 15.0, 30.0])                 # km/s, mode 0 Rayleigh
lags, acf_z = synthetic_acf(model, 0.06, preset("tauzin2019"), component="vertical")
lags, acf_p = synthetic_acf(model, 0.06, preset("tauzin2019"), component="pressure")
```

## Attenuation (`forward.attenuation`)

`LayeredModel(..., qp=..., qs=..., q_reference_frequency=1.0)` makes layers
attenuating with constant quality factors (`inf` = elastic; default reference
1 Hz, assumed). Their `vp` and `vs` are then **phase velocities at the
reference frequency**, and every solver disperses them to its own
frequencies with the constant-Q model of Kjartansson (1979):

    v~(f) = v_ref cos(pi gamma / 2) (f / f_ref)^gamma exp(-i pi gamma / 2),
    gamma = arctan(1 / Q) / pi,

so the phase velocity is v_ref (f / f_ref)^gamma, about
1 + ln(f / f_ref) / (pi Q): for Qs = 20, Vs at 0.01 Hz is 7 % below Vs at
1 Hz.

| solver | how Q enters |
|---|---|
| ACF response | complex velocities per frequency: attenuation and dispersion (eigenvectors per frequency for attenuating layers) |
| compliance | real velocities at each frequency (`dispersed`); the imaginary moduli change \|eta\| only at order 1/Q^2 |
| phase velocity | real velocities at 1 / period, one disba call per period (first-order physical dispersion) |

Without Q every solver runs its elastic code, so results are unchanged bit
for bit. Cost with an attenuating sediment (12 layers) in the synthetic
example: 24 ms per posterior evaluation against 6 ms elastic, half of it the
per-period disba calls.

Checked against: Q = Re M / -Im M and the phase-velocity power law for the
formula itself; an independent normal-incidence acoustic propagator with
complex velocities (ACF response, rtol 1e-9); the static half-space
compliance at the dispersed velocities (within the dynamic correction); the
Rayleigh speed of a homogeneous half-space with Qp = Qs, which scales exactly
by (f / f_ref)^gamma; and the elastic limit (Q = 1e12).

**What sediment Q changes** (synthetic example, 2 km water, sediment 0.6 km,
Vs 0.15-0.6 km/s, Q in the sediment only; mean (Delta / sigma)^2 per datum
for the true model, largest |Delta| / sigma in brackets, the data sigma of the
example):

| Qs | Qp | f_ref (Hz) | compliance | dispersion | ACF Z | ACF P |
|---|---|---|---|---|---|---|
| 10 | inf | 1 | 1.85 (2.4) | 0.003 (0.16) | 0.003 (0.14) | 0.002 (0.15) |
| 20 | inf | 1 | 0.41 (1.1) | 0.001 (0.07) | 0.001 (0.11) | 0.001 (0.12) |
| 50 | inf | 1 | 0.06 (0.4) | 0.000 (0.02) | 0.000 (0.06) | 0.000 (0.08) |
| 20 | 100 | 1 | 0.51 (1.2) | 0.002 (0.12) | 0.008 (0.30) | 0.007 (0.27) |
| 20 | 30 | 1 | 0.92 (1.2) | 0.006 (0.24) | 0.070 (0.82) | 0.068 (0.93) |
| 20 | 100 | 0.01 | 0.01 (0.3) | 0.001 (0.12) | 0.056 (0.96) | 0.056 (0.92) |
| 20 | 100 | 0.1 | 0.09 (0.4) | 0.000 (0.01) | 0.021 (0.59) | 0.019 (0.52) |

The ACF barely sees Qs and sees Qp only when Qp is a few tens. Most of the effect is velocity dispersion: it moves
whichever data lie far from the reference frequency, compliance for 1 Hz,
the ACF for 0.01 Hz. Q thus mainly says **at which frequency the model's
velocities hold**; an elastic model fitted to all three data types takes
velocities that compromise between the bands.

## Compliance (`forward.compliance`)

Infragravity-wave loading with wavenumber from `omega^2 = g k tanh(k H)`,
solid response from Crawford's propagator (`raydep_ft`, Fortran). Water
enters only through its depth.

- `wavenumber="exact"` (default) solves the dispersion relation by Newton
  iteration. `"poly"` is the v0.1 approximation, up to ~2 % off in k near
  k H ~ 1-2 (up to ~3 % in compliance); use it only to reproduce old results.
- For each frequency the model is cut where the loading has decayed by
  e^-25 (`decay_cutoff`); below that the propagator would overflow on deep
  models, and the neglected structure has no measurable effect.
- `compliance_kernel` gives d ln(eta) / d ln(m) per layer by central
  differences.
- `infragravity_cutoff(depth_m)`: the frequency where the wavelength equals
  the water depth, the usual upper limit of compliance data (checked: k H =
  2 pi from the exact wavenumber solver).

Checked against: the v0.1 outputs (`tests/data/legacy_v0_1_forward.npz`), and
the static half-space solution `(lambda + 2 mu) / (2 mu (lambda + mu))`, which
it approaches with a dynamic correction of order `(c / vs)^2`.

## Rayleigh phase velocity (`forward.dispersion`)

disba with the water as a fluid top layer (vs = 0). `mode=0` is the
fundamental mode. With a few km of water it drops steeply below ~10 s; that is
the physical fundamental mode, controlled by water depth and shallow sediment.

Checked against: the analytic Scholte-wave speed (deep water, short period)
for a hard and a soft solid, and the solid-only result as the water thins.

Choosing the model bottom: `phase_velocity_kernel` returns dc/dm per layer.
For a typical oceanic model the 30 s Vs kernel holds 90 % of its weight above
~100 km and 99 % above ~155 km, so the half-space should start at ~150 km (it
can be fixed to a reference model below ~100 km).

## Teleseismic ACF (`forward.response`, `forward.acf`, `acf`)

`seafloor_response` solves the plane-P-wave problem for half-space, solid
layers, water column and pressure-free sea surface, and returns vertical and
radial motion of the seafloor and the pressure just above it (vertical
positive up, pressure positive in compression, unit incident amplitude).
`synthetic_acf` passes that transfer function through an
`acf.ACFProcessing`, the same object `acf.acf_from_trace` applies to recorded
traces:

| step | data | synthetic |
|---|---|---|
| cut P-coda window, demean, detrend, taper | yes | - |
| whitening (running mean width, water level) | yes | yes |
| autocorrelation, lags 0..max_lag | yes | yes |
| zero-phase filters, chained in order (band-, high-, low-pass, Gaussian) | yes | yes |
| zero-lag sin^2 taper, normalization | yes | yes |
| stacking (linear or phase-weighted) | yes | - |

Presets live in `acf.PRESETS`:

- `tauzin2019`: the flow of Tauzin, Pham & Tkalcic (2019, GJI 217, 1-24,
  doi:10.1093/gji/ggz002); its numbers are unverified starting values
  (band-pass 0.5-4 Hz, sin^2 zero-lag taper, normalized to the maximum).
- `kim2023`: the Z-/P-ACF settings of Kim et al. (2023, JGR 128,
  e2023JB027616, Supporting Text S2): power spectrum divided by its running
  mean over W = 0.8 Hz (2N + 1 samples, W = 2N df), no water level, Gaussian
  low-pass a = 5, then a 0.5 Hz high-pass (half-power band about
  0.55-1.7 Hz), normalized at t = 0, zero lag not tapered. Only the high-pass
  order (2) is assumed. Their data window is 70 s from 10 s before P with a
  Tukey taper of cosine fraction 0.2 (`acf_from_trace(..., taper_fraction=0.2)`).

`Whitening(average="power")` divides |X|^2 by its running mean (Kim);
`average="amplitude"` divides |X| by the running mean of |X| and squares.

`synthetic_acfs(..., snr=...)` follows Kim et al. (2023, Supporting Text S4):
noise lowers the amplitude of zero-lag-normalized ACFs, so a flat noise
amplitude N = sqrt(sum |S|^2 / (L (SNR^2 - 1))) is added to the synthetic
amplitude spectrum before processing. The sum is taken over the processed band
of a unit-impulse synthetic; the paper leaves both choices open.

How well this models noisy stacks is open. In simulations (synthetic events:
the true response of the example model to random 3 s source wavelets plus
white noise, kim2023, 1000 events) the stack sat 9-17 stack sigmas (for 60
events) from the noise-free synthetic; `snr` from the broadband SNR reduced
that to 3-4, `snr` from the SNR within the processed band to 2-3, and adding
the noise in power (sqrt(|S|^2 + N^2)) instead of amplitude with the in-band
SNR to 0.5-2. Noise-free events stack to within about 1 sigma (the residual
is the source spectrum that whitening does not flatten). Measure SNR in the
processing band, and treat the amplitude of noisy ACFs with care until this is
settled.

The narrower band is what makes `kim2023` ACFs look smoother; normalizing at
t = 0 also keeps reflection amplitudes meaningful, whereas normalizing to the
maximum after tapering makes them depend on the taper.

Performance and accuracy controls:

- `synthetic_acfs` computes one seafloor response and returns the ACF of
  several components from it (use it for Z + P in an inversion).
- Frequencies where the combined filter power is below `band_tolerance`
  (default 1e-6) of its peak are dropped, for data and synthetics alike;
  `kim2023` then needs frequencies up to ~6.4 Hz only. The whitening water
  level refers to the maximum smoothed amplitude within that band.
- `duration` (default 200 s) sets the frequency spacing; the elastic response
  has no attenuation, so in deep water over a hard seafloor it is still
  ringing after 100 s. `wraparound_fraction` returns the energy fraction of the
  filtered response in the last 10 % of the window; check it before using a
  shorter window.
- `water_multiple_lags(water, ray_p, n)`: lags m 2 H sqrt(1/vp_w^2 - p^2) of
  the water reverberations, the strongest ACF arrivals (checked against the
  strongest peak of a synthetic ACF over a half-space).
- Cost for a 32-layer model, 2 km water, dt = 0.05 s, 200 s: about 4 ms for
  Z + P with `kim2023` and 8 ms with `tauzin2019` (the response is computed in
  numpy with the frequency axis folded into one matrix product per layer).

Checked against: energy-flux conservation (incident = reflected P + S, with
and without water), an independent normal-incidence acoustic transfer-matrix
solution, sediment and water reverberation times `2 H q`, the opposite
polarity of water multiples on vertical and pressure, and agreement of the
trace and spectrum processing paths on the same impulse response.

Not yet done: comparison with rein-acfcompute (GPL-3.0: run it locally as a
reference only, do not copy code); attenuation (Q) in any solver.
