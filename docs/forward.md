# Forward solvers

Three observables share one model, `complybay.model.OBSModel`: an optional
`WaterLayer` over a `LayeredModel` (n solid layers plus a half-space).
Units everywhere: km, km/s, g/cm³; depth 0 is the seafloor.

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

$$
\tilde v(f) = v_\mathrm{ref} \cos\frac{\pi\gamma}{2}
\left(\frac{f}{f_\mathrm{ref}}\right)^{\gamma} e^{-i\pi\gamma/2},
\qquad \gamma = \frac{1}{\pi}\arctan\frac{1}{Q},
$$

so the phase velocity is $v_\mathrm{ref} (f / f_\mathrm{ref})^\gamma$, relative to
$v_\mathrm{ref}$ about $1 + \ln(f / f_\mathrm{ref}) / (\pi Q)$: for $Q_s = 20$, Vs at
0.01 Hz is 7 % below Vs at 1 Hz.

| solver | how Q enters |
|---|---|
| ACF response | complex velocities per frequency: attenuation and dispersion (eigenvectors per frequency for attenuating layers) |
| compliance | real velocities at each frequency (`dispersed`); the imaginary moduli change $\lvert\eta\rvert$ only at order $1/Q^2$ |
| phase velocity | real velocities at 1 / period, one disba call per period (first-order physical dispersion) |

Without Q every solver runs its elastic code, so results are unchanged bit
for bit. Cost with an attenuating sediment (12 layers) in the synthetic
example: 24 ms per posterior evaluation against 6 ms elastic, half of it the
per-period disba calls.

Checked against: $Q = \operatorname{Re} M / (-\operatorname{Im} M)$ and the phase-velocity power law for the
formula itself; an independent normal-incidence acoustic propagator with
complex velocities (ACF response, rtol 1e-9); the static half-space
compliance at the dispersed velocities (within the dynamic correction); the
Rayleigh speed of a homogeneous half-space with Qp = Qs, which scales exactly
by $(f / f_\mathrm{ref})^\gamma$; and the elastic limit ($Q = 10^{12}$).

**What sediment Q changes** (synthetic example, 2 km water, sediment 0.6 km,
Vs 0.15-0.6 km/s, Q in the sediment only; mean $(\Delta / \sigma)^2$ per datum
for the true model, largest $\lvert\Delta\rvert / \sigma$ in brackets, the data sigma of the
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

Infragravity-wave loading with wavenumber from $\omega^2 = g k \tanh(k H)$,
solid response from Crawford's propagator (`raydep_ft`, Fortran). Water
enters only through its depth.

- `wavenumber="exact"` (default) solves the dispersion relation by Newton
  iteration. `"poly"` is the v0.1 approximation, up to ~2 % off in $k$ near
  $kH \approx 1$-2 (up to ~3 % in compliance); use it only to reproduce old results.
- For each frequency the model is cut where the loading has decayed by
  $e^{-25}$ (`decay_cutoff`); below that the propagator would overflow on deep
  models, and the neglected structure has no measurable effect.
- `compliance_kernel` gives $\partial \ln\eta / \partial \ln m$ per layer by central
  differences.
- `infragravity_cutoff(depth_m)`: the frequency where the wavelength equals
  the water depth, the usual upper limit of compliance data (checked: $kH = 2\pi$
  from the exact wavenumber solver).

Checked against: the v0.1 outputs (`tests/data/legacy_v0_1_forward.npz`), and
the static half-space solution

$$
\eta_0 = \frac{\lambda + 2\mu}{2\mu(\lambda + \mu)},
$$

which it approaches with a dynamic correction of order $(c / v_s)^2$.

## Rayleigh phase velocity (`forward.dispersion`)

disba with the water as a fluid top layer (vs = 0). `mode=0` is the
fundamental mode. With a few km of water it drops steeply below ~10 s; that is
the physical fundamental mode, controlled by water depth and shallow sediment.

Checked against: the analytic Scholte-wave speed (deep water, short period)
for a hard and a soft solid, and the solid-only result as the water thins.

Choosing the model bottom: `phase_velocity_kernel` returns $\partial c / \partial m$ per layer.
For a typical oceanic model the 30 s Vs kernel holds 90 % of its weight above
~100 km and 99 % above ~155 km, so the half-space should start at ~150 km (it
can be fixed to a reference model below ~100 km).

## Teleseismic ACF (`forward.response`, `forward.acf`, `acf`)

A synthetic ACF is computed in two steps, both in the frequency domain:

```text
model, ray_p ──seafloor_response──> H(f): transfer function of Z, P (and R) at the seafloor
H(f) ──ACFProcessing (same as for data)──> ACF(τ), lags 0..max_lag
```

### Why the ACF of the transfer function

A recorded P coda is the source wavelet convolved with the structure's
impulse response, $X(f) = S(f)\,H(f)$, so its autocorrelation has the power
spectrum $\lvert S\rvert^2 \lvert H\rvert^2$. Whitening divides by a smoothed
power spectrum, which removes the smooth $\lvert S\rvert^2$ of a short source
(and noise level) but keeps the ripple of $\lvert H\rvert^2$ that the
reverberations make. A reverberation of delay $\tau$ adds a term
$\varepsilon e^{2\pi i f \tau}$ to $H$ and hence a ripple
$2\varepsilon \cos(2\pi f \tau)$ to $\lvert H\rvert^2$, which the inverse FFT
turns into a peak at lag $\tau$. The synthetic therefore takes a unit-impulse
source, $S = 1$, and pushes $H$ through the same processing as the data.

### Step 1: seafloor transfer function (`forward.response`)

A plane P wave of horizontal slowness $p$ (`ray_p`, s/km) and unit
displacement amplitude comes up from the half-space. Time dependence is
$e^{-i\omega t}$, $z$ points down, and every layer is described by the
motion-stress vector

$$
\mathbf v = (u_x,\ u_z,\ \tau_{xz},\ \tau_{zz}),
\qquad \tau = \sigma / (i\omega),
$$

scaled so that the layer matrices below do not depend on frequency (elastic
layers).

1. **One layer.** In a homogeneous layer $\mathbf v$ is a sum of four plane
    waves (down P, down S, up P, up S) with vertical slownesses
    $\pm q_\alpha$, $\pm q_\beta$, $q = \sqrt{1/v^2 - p^2}$ (principal root, so
    evanescent or attenuated downgoing waves decay with depth). With $E$ the
    matrix of their polarization vectors, a layer of thickness $h$ carries
    $\mathbf v$ from its top to its bottom by

    $$
    P(h) = E\, \operatorname{diag}\!\left(e^{i\omega q_j h}\right) E^{-1}.
    $$

2. **Seafloor unknowns.** Below water (or vacuum) the shear traction vanishes,
    $\tau_{xz}(0) = 0$, so the seafloor state has three unknowns
    $\mathbf x = (u_x, u_z, \tau_{zz})$; $u_x$ is free because the fluid may
    slip. The three corresponding basis vectors are carried through all solid
    layers, $P_N \cdots P_1$, and projected on the half-space waves,
    $\mathbf a = E_\mathrm{hs}^{-1} P_N \cdots P_1 B\, \mathbf x$, with $B$ the
    $4 \times 3$ matrix that places $\mathbf x$ into $\mathbf v$ and
    $\mathbf a$ the amplitudes of down P, down S, up P, up S.
3. **Three conditions** fix $\mathbf x$ at every frequency:
    - incident up-going P has unit amplitude, $a_\mathrm{P\uparrow} = 1$;
    - no up-going S comes from below, $a_\mathrm{S\uparrow} = 0$;
    - the sea surface is pressure free: $(u_z, \tau_{zz})$ is continuous at the
      seafloor, is carried up through the water with the fluid propagator
      (one P wave pair, $q_w = \sqrt{1/v_{p,w}^2 - p^2}$), and $\tau_{zz}$ must
      vanish at the top. Without water: $\tau_{zz}(0) = 0$ (land station).
4. **Outputs**, from $\mathbf x$: vertical ground velocity
    $Z = -i\omega\,(-u_z)$ (positive up), radial velocity $-i\omega\, u_x$, and
    pressure just above the seafloor $P = -\sigma_{zz} = -i\omega\,\tau_{zz}$
    (positive in compression). The down-going half-space amplitudes are kept as
    `reflected_p`, `reflected_s` for the energy check.

The frequency axis is folded into the matrix products (one $4 \times 4$
times $4 \times 3n_f$ product per layer), which is why a 32-layer response
costs a few ms. Attenuating layers use the complex velocities of
`forward.attenuation`, so their $E$ is built per frequency.

### Step 2: from transfer function to ACF (`forward.acf`, `acf.processing`)

`synthetic_acfs(model, ray_p, processing, dt=..., duration=...)`:

1. **FFT grid.** $n = $ `duration` / `dt` samples, frequencies
    $f_k = k / \mathrm{duration}$. The response is computed only up to the
    frequency where the filters pass less than `band_tolerance` of their peak
    power (plus half a whitening window); above that the spectrum is set to 0,
    for data as well.
2. **Amplitude, optional noise level.** $A = \lvert H\rvert$; with `snr`,
    a flat noise amplitude is added (below).
3. **Whitening** (`Whitening`). With $\overline{\,\cdot\,}_W$ a running mean
    over $W$ Hz ($2N+1$ samples, $W = 2N\Delta f$) and a floor at
    `water_level` times its maximum:

    $$
    \text{power: } \Phi = \frac{A^2}{\overline{A^2}_W},
    \qquad
    \text{amplitude: } \Phi = \left(\frac{A}{\overline{A}_W}\right)^2 .
    $$

4. **Filters**, zero phase: $\Phi \leftarrow \Phi \prod_j \lvert F_j(f)\rvert^2$
    (Butterworth band-, high- and low-pass; Gaussian with
    $\lvert F\rvert^2 = e^{-(\omega/2a)^2}$), equivalent to filtering the ACF forward and
    backward.
5. **Autocorrelation**: $\mathrm{ACF}(\tau) = \mathcal F^{-1}[\Phi]$ (a real,
    even spectrum, so no sign convention enters), kept for
    $0 \le \tau \le$ `max_lag`.
6. **Zero-lag taper** $\sin^2(\pi\tau / 2T)$ for $\tau < T$
    (`zero_lag_taper`), then **normalization** by the maximum (`max`) or by
    the zero lag (`zero_lag`, so values are fractions of the total energy).

`acf_from_trace` does the same for a recorded trace after cutting,
demeaning, detrending, tapering and zero padding; the table below lists
which steps run where.

### What whitening does to short lags

Dividing by a running mean also removes part of the structure's own ripple.
To first order in the reverberation amplitude, whitening over $W$ Hz
multiplies the ACF at lag $\tau$ by

$$
1 - \operatorname{sinc}(W\tau),
\qquad \operatorname{sinc} x = \frac{\sin \pi x}{\pi x},
$$

(checked on a single echo with power averaging: within 0.015 of the computed
ratio for $\tau = 0.3$-5 s, $W = 0.5$ and 0.8 Hz; amplitude averaging gives
the same factor to first order). Lags shorter than about $1/W$ are
damped: with `kim2023` ($W = 0.8$ Hz) an echo at 0.3 s keeps 9 % of its
amplitude and one at 0.7 s 44 %, while lags beyond 1.25 s pass (within a few
% to 15 %, the sinc side lobes). With `tauzin2019` ($W = 0.5$ Hz) the cut is
near 2 s. Shallow-sediment P reverberations (well under 1 s) are thus
weakened before any filter acts; this is the same factor that shapes the
error correlation (`docs/inference.md`).

### Reading an ACF

Peaks sit at the two-way vertical times of the reverberations and of their
combinations; the water multiples are the strongest. Example: the synthetic
model (2 km water, $v_{p,w} = 1.5$ km/s; sediment 0.6 km, Vp 1.75 km/s,
Vs 0.15-0.6 km/s) at $p = 0.06$ s/km, peaks of the noise-free ACFs:

| arrival | predicted lag (s) | Z: tauzin2019 / kim2023 | P: tauzin2019 / kim2023 |
|---|---|---|---|
| sediment P reverberation $2hq_\alpha$ | 0.68 | −0.41 / – | – / – |
| water multiple $2Hq_w$ | 2.66 | +0.64 / +0.24 | −1.00 / −0.61 |
| water + sediment | 3.34 | −1.00 / −0.39 | −0.71 / −0.34 |
| water + 2 × sediment | 4.02 | +0.42 / +0.27 | – / – |
| 2nd water multiple $4Hq_w$ | 5.31 | – / −0.15 | +0.43 / – |
| 2 × water + sediment | 5.99 | – / – | +0.78 / +0.45 |

(–: not among the eight largest peaks after 0.25 s.) The water multiple has
opposite polarity on Z and P because the pressure-free sea surface reflects
pressure with $-1$ but vertical motion with $+1$. The sediment S time
($2hq_\beta = 3.70$ s) gives no separate peak, in line with the ACF's weak
sensitivity to sediment Vs (`docs/synthetic_tests.md`, test 1), and the Moho
P reverberation (2.74 s) overlaps the water multiple. The sediment shows up
mostly as the 0.68 s it adds to each water multiple, which is why the ACF
constrains `tp2` well.

### Processing steps for data and synthetics

| step | data | synthetic |
|---|---|---|
| cut P-coda window, demean, detrend, taper | yes | - |
| whitening (running mean width, water level) | yes | yes |
| autocorrelation, lags 0..max_lag | yes | yes |
| zero-phase filters, chained in order (band-, high-, low-pass, Gaussian) | yes | yes |
| zero-lag $\sin^2$ taper, normalization | yes | yes |
| stacking (linear or phase-weighted) | yes | - |

Presets live in `acf.PRESETS`:

- `tauzin2019`: the flow of Tauzin, Pham & Tkalcic (2019, GJI 217, 1-24,
  doi:10.1093/gji/ggz002); its numbers are unverified starting values
  (band-pass 0.5-4 Hz, $\sin^2$ zero-lag taper, normalized to the maximum).
- `kim2023`: the Z-/P-ACF settings of Kim et al. (2023, JGR 128,
  e2023JB027616, Supporting Text S2): power spectrum divided by its running
  mean over $W = 0.8$ Hz ($2N + 1$ samples, $W = 2N\,\Delta f$), no water level, Gaussian
  low-pass $a = 5$, then a 0.5 Hz high-pass (half-power band about
  0.55-1.7 Hz), normalized at $t = 0$, zero lag not tapered. Only the high-pass
  order (2) is assumed. Their data window is 70 s from 10 s before P with a
  Tukey taper of cosine fraction 0.2 (`acf_from_trace(..., taper_fraction=0.2)`).

`Whitening(average="power")` divides $\lvert X\rvert^2$ by its running mean (Kim);
`average="amplitude"` divides $\lvert X\rvert$ by the running mean of $\lvert X\rvert$ and squares.

`synthetic_acfs(..., snr=...)` follows Kim et al. (2023, Supporting Text S4):
noise lowers the amplitude of zero-lag-normalized ACFs, so a flat noise
amplitude

$$
N = \sqrt{\frac{\sum \lvert S\rvert^2}{L\,(\mathrm{SNR}^2 - 1)}}
$$

is added to the synthetic amplitude spectrum before processing. The sum is taken over the processed band
of a unit-impulse synthetic; the paper leaves both choices open.

How well this models noisy stacks is open. In simulations (synthetic events:
the true response of the example model to random 3 s source wavelets plus
white noise, kim2023, 1000 events) the stack sat 9-17 stack sigmas (for 60
events) from the noise-free synthetic; `snr` from the broadband SNR reduced
that to 3-4, `snr` from the SNR within the processed band to 2-3, and adding
the noise in power ($\sqrt{\lvert S\rvert^2 + N^2}$) instead of amplitude with the in-band
SNR to 0.5-2. Noise-free events stack to within about 1 sigma (the residual
is the source spectrum that whitening does not flatten). Measure SNR in the
processing band, and treat the amplitude of noisy ACFs with care until this is
settled.

The narrower band is what makes `kim2023` ACFs look smoother; normalizing at
$t = 0$ also keeps reflection amplitudes meaningful, whereas normalizing to the
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
- `water_multiple_lags(water, ray_p, n)`: lags $2mH\sqrt{1/v_{p,w}^2 - p^2}$ of
  the water reverberations, the strongest ACF arrivals (checked against the
  strongest peak of a synthetic ACF over a half-space).
- Cost for a 32-layer model, 2 km water, $\Delta t = 0.05$ s, 200 s: about 4 ms for
  Z + P with `kim2023` and 8 ms with `tauzin2019` (the response is computed in
  numpy with the frequency axis folded into one matrix product per layer).

Checked against: energy-flux conservation (incident = reflected P + S, with
and without water), an independent normal-incidence acoustic transfer-matrix
solution, sediment and water reverberation times $2Hq$, the opposite
polarity of water multiples on vertical and pressure, and agreement of the
trace and spectrum processing paths on the same impulse response.

Not yet done: comparison with rein-acfcompute (GPL-3.0: run it locally as a
reference only, do not copy code).
