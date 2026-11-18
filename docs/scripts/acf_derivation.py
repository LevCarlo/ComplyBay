"""Formulas and tables on docs/acf.md, checked against the package with numpy.

Writes docs/generated/acf_*.md, which the page includes, and asserts:

- the seafloor outputs satisfy the pressure-free sea surface, checked with
  an independent fluid solution (Helmholtz equation for the pressure in the
  water), with and without attenuation, at oblique incidence;
- a land half-space doubles the vertical motion at normal incidence;
- the whitening formulas, against ``Whitening.whitened_power`` with an
  explicit running mean;
- the noise level of ``add_noise_level`` against Parseval in the time domain;
- tables: the frequency limit of each preset and the wraparound fraction
  against the window length (measured, not asserted beyond monotonicity).

Run: python docs/scripts/acf_derivation.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from _models import SEDIMENT_OVER_CRUST, sediment_over_crust
from _style import GENERATED_DIR
from scipy.fft import irfft, next_fast_len, rfftfreq

from complybay.acf import Whitening, preset
from complybay.forward.acf import add_noise_level, wraparound_fraction
from complybay.forward.response import _solid_eigenvectors, seafloor_response
from complybay.model import LayeredModel, OBSModel, WaterLayer


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"acf_{name}.md").write_text(text, encoding="utf-8")


# --- 1. Seafloor outputs and the sea-surface condition -----------------------------------
_write(
    "outputs",
    "$$\n\\mathbf v = (u_x,\\ u_z,\\ \\tau_{xz},\\ \\tau_{zz}),\\qquad"
    " \\tau = \\frac{\\sigma}{i\\omega},\\qquad z\\ \\text{down},\\qquad e^{-i\\omega t}\n$$\n\n"
    "$$\n\\boxed{\\;Z = -i\\omega\\,(-u_z),\\qquad R = -i\\omega\\,u_x,"
    "\\qquad P = -\\sigma_{zz} = -i\\omega\\,\\tau_{zz}\\;}\n$$\n",
)
_write(
    "sea_surface",
    "$$\nP(z) = P_0\\cos(k_z z) + \\frac{\\rho_w\\omega^2 u_z(0)}{k_z}\\sin(k_z z),"
    "\\qquad k_z = \\omega q_w,\\quad q_w = \\sqrt{\\frac{1}{v_{p,w}^2} - p^2}\n$$\n\n"
    "$$\nP(-H) = 0 \\;\\Longleftrightarrow\\; P_0\\cos(\\omega q_w H)"
    " = \\frac{\\rho_w\\,\\omega\\,u_z(0)}{q_w}\\,\\sin(\\omega q_w H)\n$$\n",
)
freqs = np.linspace(0.01, 6.0, 600)
omega = 2 * np.pi * freqs
attenuating = replace(
    SEDIMENT_OVER_CRUST,
    qp=np.r_[50.0, np.full(SEDIMENT_OVER_CRUST.n_layers, np.inf)],
    qs=np.r_[15.0, np.full(SEDIMENT_OVER_CRUST.n_layers, np.inf)],
)
worst_surface = 0.0
for solid in (SEDIMENT_OVER_CRUST, attenuating):
    for depth_km, ray_p in ((2.0, 0.06), (4.5, 0.03), (0.3, 0.0)):
        water = WaterLayer(depth_km)
        r = seafloor_response(OBSModel(solid, water), ray_p, freqs)
        u_down = -r.displacement_z  # the response returns u_z positive up
        q_w = np.sqrt(1 / water.vp**2 - ray_p**2)
        lhs = r.pressure * np.cos(omega * q_w * water.thickness)
        rhs = water.rho * omega * u_down / q_w * np.sin(omega * q_w * water.thickness)
        scale = np.abs(r.pressure) + np.abs(water.rho * omega * u_down / q_w)
        worst_surface = max(worst_surface, np.max(np.abs(lhs - rhs) / scale))
# Independent solution of the same boundary-value problem: round-off of the 4x4
# propagator products and solves, 1e-10.
assert worst_surface < 1e-10, worst_surface
land = seafloor_response(OBSModel(LayeredModel([], [6.0], [3.5], [2.8])), 0.0, freqs)
assert np.allclose(land.displacement_z, 2.0, rtol=1e-12)  # free-surface doubling
assert np.allclose(land.pressure, 0.0)

# Layer eigenvectors (appendix): columns are plane waves u = U exp(i omega (p x + q z - t))
# of unit displacement; their traction rows are sigma / (i omega) from Hooke's law. Checked
# by differentiating the plane wave numerically (central differences, step 1e-6 of the
# wavelength: truncation ~1e-12, round-off ~1e-9 relative; tolerance 1e-7).
_write(
    "propagator",
    "$$\nq_\\alpha = \\sqrt{\\frac{1}{v_p^2} - p^2},\\qquad"
    " q_\\beta = \\sqrt{\\frac{1}{v_s^2} - p^2}\\qquad(\\operatorname{Im} q \\ge 0)\n$$\n\n"
    "$$\n\\text{P}: (u_x, u_z) = v_p\\,(p,\\ \\pm q_\\alpha),\\qquad"
    " \\text{S}: (u_x, u_z) = v_s\\,(\\pm q_\\beta,\\ -p)\n$$\n\n"
    "$$\n\\tau_{xz} = \\mu\\,(p\\,u_z + q\\,u_x),\\qquad"
    " \\tau_{zz} = \\lambda\\,(p\\,u_x + q\\,u_z) + 2\\mu\\, q\\, u_z\n$$\n\n"
    "$$\n\\boxed{\\;\\mathbf v(z + h) = E\\,"
    "\\operatorname{diag}\\!\\left(e^{i\\omega q_j h}\\right) E^{-1}\\,\\mathbf v(z)\\;}"
    "\\qquad q_j = (q_\\alpha, q_\\beta, -q_\\alpha, -q_\\beta)\n$$\n",
)
vp, vs, rho, ray_p, w = 6.0, 3.5, 2.8, 0.06, 2 * np.pi * 1.3
mu = rho * vs**2
lam = rho * vp**2 - 2 * mu
q_a, q_b = np.sqrt(1 / vp**2 - ray_p**2), np.sqrt(1 / vs**2 - ray_p**2)
columns = [
    (q_a, (ray_p * vp, q_a * vp)),
    (q_b, (q_b * vs, -ray_p * vs)),
    (-q_a, (ray_p * vp, -q_a * vp)),
    (-q_b, (-q_b * vs, -ray_p * vs)),
]
step = 1e-6 / (w * q_a)
for q, (ux, uz) in columns:

    def u(x, z, ux=ux, uz=uz, q=q):
        return np.array([ux, uz]) * np.exp(1j * w * (ray_p * x + q * z))

    du_dx = (u(step, 0) - u(-step, 0)) / (2 * step)
    du_dz = (u(0, step) - u(0, -step)) / (2 * step)
    sigma_xz = mu * (du_dx[1] + du_dz[0])
    sigma_zz = lam * (du_dx[0] + du_dz[1]) + 2 * mu * du_dz[1]
    tau = np.array([mu * (ray_p * uz + q * ux), lam * (ray_p * ux + q * uz) + 2 * mu * q * uz])
    assert np.allclose(np.array([sigma_xz, sigma_zz]) / (1j * w), tau, rtol=1e-7)
    assert np.isclose(abs(ux) ** 2 + abs(uz) ** 2, 1.0)  # unit displacement amplitude
    # Equation of motion: rho omega^2 u = -div sigma for this plane wave, i.e. the
    # slowness vector (p, q) has length 1 / v of its wave type.
    v = vp if abs(abs(q) - q_a) < 1e-12 else vs
    assert np.isclose(ray_p**2 + q**2, 1 / v**2)
# The same columns as the package's layer matrix (private helper; same formula, round-off).
eigvecs, slowness = _solid_eigenvectors(vp, vs, rho, ray_p)
for j, (q, (ux, uz)) in enumerate(columns):
    tau = [mu * (ray_p * uz + q * ux), lam * (ray_p * ux + q * uz) + 2 * mu * q * uz]
    assert np.allclose(eigvecs[:, j], [ux, uz, *tau], rtol=1e-12) and np.isclose(slowness[j], q)

# --- 2. Whitening ----------------------------------------------------------------------------
_write(
    "whitening",
    "$$\n\\text{power: } \\Phi = \\frac{A^2}{\\max\\!\\left(\\overline{A^2}_W,\\,"
    " \\ell \\max_\\mathrm{band}\\overline{A^2}_W\\right)},"
    "\\qquad\n\\text{amplitude: } \\Phi = \\left(\\frac{A}{\\max\\!\\left(\\overline{A}_W,\\,"
    " \\ell \\max_\\mathrm{band}\\overline{A}_W\\right)}\\right)^{2}\n$$\n\n"
    "with $A = \\lvert H\\rvert$, $\\overline{\\,\\cdot\\,}_W$ the mean over $2N + 1$"
    " frequency samples, $N = \\mathrm{round}(W / 2\\Delta f)$, and $\\ell$ = `water_level`.\n",
)
rng = np.random.default_rng(0)
amplitude = rng.uniform(0.2, 3.0, 1000)
df, band = 0.005, np.ones(1000, bool)
for average in ("power", "amplitude"):
    for width, level in ((0.8, 0.0), (0.5, 0.01), (0.5, 0.9)):
        n = int(round(width / (2 * df)))
        values = amplitude**2 if average == "power" else amplitude
        # Running mean of 2N + 1 samples, edges padded with the end values.
        padded = np.pad(values, n, mode="edge")
        mean = np.array([padded[i : i + 2 * n + 1].mean() for i in range(values.size)])
        expected = values / np.maximum(mean, level * mean.max())
        if average == "amplitude":
            expected = expected**2
        got = Whitening(width, level, average).whitened_power(amplitude, df, band)
        # Different summation order: round-off, 1e-10.
        assert np.allclose(got, expected, rtol=1e-10), (average, width, level)
        if level == 0.9:
            assert np.any(mean < level * mean.max())  # the floor is active in this case

# --- 3. Noise level ----------------------------------------------------------------------------
_write(
    "noise",
    "$$\n\\boxed{\\;\\lvert S\\rvert \\;\\to\\; \\lvert S\\rvert + N,\\qquad"
    " N = \\sqrt{\\frac{\\sum_k \\lvert S_k\\rvert^2}{L\\,(\\mathrm{SNR}^2 - 1)}}\\;}"
    "\\qquad \\mathrm{SNR}^2 = \\frac{\\sum (s^2 + n^2)}{\\sum n^2}\n$$\n\n"
    "with the sum over the two-sided DFT of length $L$ inside the processed band.\n",
)
dt, duration = 0.05, 200.0
n_fft = next_fast_len(int(round(duration / dt)), real=True)
grid = rfftfreq(n_fft, dt)
kim = preset("kim2023")
in_band = grid <= kim.frequency_limit(grid)
spectrum = np.zeros(grid.size, complex)
spectrum[in_band] = seafloor_response(sediment_over_crust(), 0.06, grid[in_band]).component(
    "vertical"
)
signal = irfft(spectrum, n=n_fft)
for snr in (1.5, 3.0, 10.0):
    noisy = add_noise_level(spectrum, n_fft, snr)
    noise = noisy[in_band][1] - np.abs(spectrum[in_band][1])
    # White noise of variance sigma^2 has E|N_k|^2 = L sigma^2 per DFT bin, and
    # sum s^2 = (SNR^2 - 1) L sigma^2 (Parseval), so N = sqrt(sum s^2 / (SNR^2 - 1)).
    expected = np.sqrt(np.sum(signal**2) / (snr**2 - 1))
    assert np.isclose(noise, expected, rtol=1e-10), (snr, noise, expected)
    # Added only where the spectrum is non-zero: inside the band, except f = 0 (Z ~ omega).
    nonzero = spectrum != 0
    assert np.array_equal(nonzero, in_band & (grid > 0))
    assert np.allclose(noisy[nonzero], np.abs(spectrum[nonzero]) + noise, rtol=1e-12)
    assert np.all(noisy[~nonzero] == 0)

# --- 4. Frequency limit and wraparound ----------------------------------------------------
limits = {name: preset(name).frequency_limit(grid) for name in ("kim2023", "tauzin2019")}
_write(
    "band",
    "| preset | filters | frequency limit (Hz), `band_tolerance` = 1e-6 |\n|---|---|---|\n"
    f"| `kim2023` | Gaussian $a$ = 5, high-pass 0.5 Hz | {limits['kim2023']:.2f} |\n"
    f"| `tauzin2019` | band-pass 0.5-4 Hz | {limits['tauzin2019']:.2f} |\n",
)
# Assumed hard seafloor: 6 km of basaltic crust over mantle. A half-space directly at the
# seafloor is not used: the phase reference is the top of the half-space, so its direct
# arrival sits at t = 0 and half of the zero-phase-filtered pulse lands at the end of the
# window, whatever the duration (fraction ~0.5).
hard = LayeredModel([6.0], [6.0, 8.1], [3.5, 4.5], [2.8, 3.35])
cases = {
    "docs model, 2 km water": sediment_over_crust(2.0),
    "docs model, 5 km water": sediment_over_crust(5.0),
    "hard crust, 5 km water": OBSModel(hard, WaterLayer(5.0)),
}
half_space = OBSModel(LayeredModel([], [6.0], [3.5], [2.8]), WaterLayer(5.0))
assert all(wraparound_fraction(half_space, 0.06, kim, duration=d) > 0.4 for d in (60.0, 200.0))
# The same precursor sets a floor of ~5e-4 for the 6 km crust (first arrival ~0.9 s after
# t = 0): doubling the window to 400 s barely changes it, while with a 20 km crust (first
# arrival ~3 s) the fraction keeps falling.
floor = [
    wraparound_fraction(OBSModel(hard, WaterLayer(5.0)), 0.06, kim, duration=d)
    for d in (200.0, 400.0)
]
assert floor[1] > 0.5 * floor[0], floor
deep = LayeredModel([20.0], [6.0, 8.1], [3.5, 4.5], [2.8, 3.35])
deep_fraction = [
    wraparound_fraction(OBSModel(deep, WaterLayer(5.0)), 0.06, kim, duration=d)
    for d in (200.0, 400.0)
]
assert deep_fraction[1] < 0.01 * deep_fraction[0], deep_fraction
durations = (60.0, 100.0, 200.0)
rows = []
for label, model in cases.items():
    for name in ("kim2023", "tauzin2019"):
        fractions = [wraparound_fraction(model, 0.06, preset(name), duration=d) for d in durations]
        assert fractions[-1] < fractions[0], (label, name, fractions)
        rows.append(f"| {label} | `{name}` | " + " | ".join(f"{x:.0e}" for x in fractions) + " |")
_write(
    "wraparound",
    "| model | preset | "
    + " | ".join(f"{d:g} s" for d in durations)
    + " |\n|---|---|"
    + "---|" * len(durations)
    + "\n"
    + "\n".join(rows)
    + "\n",
)
print(
    f"wrote {GENERATED_DIR}/acf_*.md; sea-surface residual {worst_surface:.1e}; "
    f"frequency limits {limits}"
)
print("\n".join(rows))
