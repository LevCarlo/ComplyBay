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
  e2023JB027616): Gaussian low-pass a = 5 followed by a 0.5 Hz high-pass
  (half-power band about 0.55-1.7 Hz), smoothing width 0.8, normalized at
  t = 0, zero lag not tapered. The smoothing-width unit (taken as Hz), water
  level and high-pass order are assumptions until checked against their
  Supporting Text S2.

The narrower band is what makes `kim2023` ACFs look smoother; normalizing at
t = 0 also keeps reflection amplitudes meaningful, whereas normalizing to the
maximum after tapering makes them depend on the taper.

Checked against: energy-flux conservation (incident = reflected P + S, with
and without water), an independent normal-incidence acoustic transfer-matrix
solution, sediment and water reverberation times `2 H q`, the opposite
polarity of water multiples on vertical and pressure, and agreement of the
trace and spectrum processing paths on the same impulse response.

Not yet done: comparison with rein-acfcompute (GPL-3.0: run it locally as a
reference only, do not copy code) and checking the `kim2023` assumptions against their Supporting Text S2;
attenuation (Q) in any solver.
