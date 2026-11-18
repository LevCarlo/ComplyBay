"""Formulas on docs/likelihood.md, checked against the package with numpy.

Writes LaTeX snippets docs/generated/likelihood_*.md, which the page includes.
Each formula is written once here, evaluated with numpy, and asserted against
the package (tolerance and reason next to each):

- the Gaussian log likelihood of one data group with a noise scale h, and its
  best h, against ``JointLikelihood.evaluate``;
- the reduced-rank block likelihood, against ``ErrorCorrelation.log_likelihood``
  (independent evaluation with ``numpy.linalg.solve`` / ``slogdet``), and the
  full-rank block against scipy's multivariate normal;
- the processing-theory correlation of ACF errors, against
  ``processing_error_correlation`` (a loop over the raw lags here, the
  package builds a vectorized basis);
- the shrinkage weight, against ``shrink_correlation`` (the products
  w_kij formed explicitly here);
- the effective number of data.

Run: python docs/scripts/likelihood_formulas.py
"""

from __future__ import annotations

import numpy as np
from _models import sediment_over_crust
from _style import GENERATED_DIR
from scipy.stats import multivariate_normal

from complybay.acf import preset, processing_error_correlation, shrink_correlation
from complybay.forward import compliance
from complybay.inference import ComplianceObservation, ErrorCorrelation, JointLikelihood


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"likelihood_{name}.md").write_text(text, encoding="utf-8")


rng = np.random.default_rng(0)

# --- 1. One group with a noise scale -----------------------------------------------------------
_write(
    "group",
    "$$\n\\boxed{\\;\\ln L_k = -\\frac{1}{2} \\sum_{i=1}^{N_k} "
    "\\left(\\frac{d_i - g_i(\\mathbf m)}{h_k\\, s_i}\\right)^2 - N_k \\ln h_k\\;}"
    "\\qquad \\Rightarrow\\qquad \\hat h_k^2 = \\frac{\\chi^2_k}{N_k},\\quad "
    "\\chi^2_k = \\sum_i \\left(\\frac{d_i - g_i}{s_i}\\right)^2\n$$\n",
)
model = sediment_over_crust()
freqs = np.geomspace(0.004, 0.025, 12)
eta = compliance(model, freqs)
sigma = 0.1 * eta
data = eta * (1 + 0.17 * rng.standard_normal(freqs.size))  # residuals ~1.7 sigma
likelihood = JointLikelihood((ComplianceObservation(freqs, data, sigma, noise_parameter="h"),))
chi2 = float(np.sum(((data - eta) / sigma) ** 2))
for log10_h in (-0.4, 0.0, 0.3):
    h = 10**log10_h
    expected = -0.5 * chi2 / h**2 - freqs.size * np.log(h)
    got = likelihood.evaluate(model, {"h": log10_h}).log_likelihood
    assert np.isclose(got, expected, rtol=1e-12)  # same arithmetic
grid = np.linspace(-0.5, 0.8, 13001)
values = [likelihood.evaluate(model, {"h": g}).log_likelihood for g in grid[::50]]
best = grid[::50][int(np.argmax(values))]
fine = grid[(grid > best - 0.01) & (grid < best + 0.01)]
best = fine[np.argmax([likelihood.evaluate(model, {"h": g}).log_likelihood for g in fine])]
assert np.isclose(10 ** (2 * best), chi2 / freqs.size, rtol=1e-3)  # grid step 1e-4 in log10

# --- 2. Reduced-rank correlated block ----------------------------------------------------------
_write(
    "block",
    "$$\n\\mathbf z = \\left(\\frac{d_i - g_i}{s_i}\\right)_i,\\quad "
    "\\mathbf R = \\mathbf V \\boldsymbol\\Lambda \\mathbf V^\\mathsf{T},\\quad "
    "\\mathbf Q = \\mathbf V_{k} \\boldsymbol\\Lambda_{k}^{-1/2},\\quad "
    "k = \\min\\Big\\{k : \\sum_{i \\le k} \\lambda_i \\ge f \\sum_i \\lambda_i\\Big\\}\n$$\n\n"
    "$$\n\\boxed{\\;\\ln L_\\mathrm{block} = -\\frac12\\, \\mathbf y^\\mathsf{T} \\mathbf M^{-1} "
    "\\mathbf y - \\frac12 \\ln\\det \\mathbf M,\\qquad \\mathbf y = \\mathbf Q^\\mathsf{T} "
    "\\mathbf z,\\quad \\mathbf M = \\mathbf Q^\\mathsf{T} \\mathbf D \\mathbf R \\mathbf D "
    "\\mathbf Q\\;}\n$$\n\n"
    "$$\n\\mathbf D = \\operatorname{diag}(h \\text{ of the member owning each sample});\\qquad "
    "\\text{one } h:\\quad \\ln L_\\mathrm{block} = -\\frac{\\lvert \\mathbf y \\rvert^2}{2h^2}"
    " - k \\ln h\n$$\n",
)
p = 40
ar = 0.8 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
corr = np.block([[ar, 0.4 * ar], [0.4 * ar, ar]])  # two members, cross-correlated
z = np.linalg.cholesky(corr) @ rng.standard_normal(2 * p)
for fraction in (0.9, 0.99, 1.0):
    block = ErrorCorrelation(("a", "b"), corr, variance_fraction=fraction)
    lam, vec = np.linalg.eigh(corr)
    lam, vec = lam[::-1], vec[:, ::-1]
    k = int(np.argmax(np.cumsum(lam) / lam.sum() >= fraction - 1e-12)) + 1
    assert block.k == k
    q = vec[:, :k] / np.sqrt(lam[:k])
    for h_a, h_b in ((1.0, 1.0), (1.3, 0.7)):
        d = np.r_[np.full(p, h_a), np.full(p, h_b)]
        m = q.T @ (d[:, None] * corr * d[None, :]) @ q
        y = q.T @ z
        expected = -0.5 * y @ np.linalg.solve(m, y) - 0.5 * np.linalg.slogdet(m)[1]
        got = block.log_likelihood(z, d)[0]
        assert np.isclose(got, expected, rtol=1e-9)  # Cholesky vs LU: round-off only
        if h_a == h_b:
            assert np.isclose(got, -0.5 * y @ y / h_a**2 - k * np.log(h_a), rtol=1e-9)
    if fraction == 1.0:
        # Full rank: the Gaussian with covariance D R D up to a constant (difference of two h).
        def full(h_a, h_b):
            d = np.r_[np.full(p, h_a), np.full(p, h_b)]
            return multivariate_normal(cov=d[:, None] * corr * d[None, :]).logpdf(z)

        diff = (
            block.log_likelihood(z, np.r_[np.full(p, 1.3), np.full(p, 0.7)])[0]
            - (block.log_likelihood(z, np.ones(2 * p))[0])
        )
        assert np.isclose(diff, full(1.3, 0.7) - full(1.0, 1.0), rtol=1e-8)

_write(
    "neff",
    "$$\n\\boxed{\\;N_\\mathrm{eff} = \\frac{(\\operatorname{tr} \\mathbf R)^2}"
    "{\\sum_{ij} R_{ij}^2}\\;}\n$$\n",
)
assert np.isclose(np.trace(np.eye(7)) ** 2 / np.sum(np.eye(7) ** 2), 7)  # independent: N
assert np.isclose(np.trace(np.ones((7, 7))) ** 2 / np.sum(np.ones((7, 7)) ** 2), 1)  # one datum

# --- 3. Processing theory of ACF errors --------------------------------------------------------
_write(
    "theory",
    "$$\n\\boxed{\\;\\varepsilon(\\tau) = T(\\tau) \\sum_{k=-\\infty}^{\\infty} "
    "g(\\tau - k\\Delta t)\\, \\big(1 - \\operatorname{sinc}(W k \\Delta t)\\big)\\, e_{\\lvert k "
    "\\rvert},\\qquad \\varepsilon_\\mathrm{n}(\\tau) = \\varepsilon(\\tau) - m(\\tau)\\, "
    "\\varepsilon(\\tau_0)\\;}\n$$\n\n"
    "$$\n\\operatorname{Var} e_0 = 2,\\ \\operatorname{Var} e_{k>0} = 1 "
    "\\text{ (independent)},\\quad "
    "g = \\mathcal F^{-1}\\Big[\\prod \\lvert H_\\mathrm{filter}(f) \\rvert^2\\Big],\\quad "
    "\\operatorname{sinc} x = \\frac{\\sin \\pi x}{\\pi x}\n$$\n",
)
kim = preset("kim2023")
dt = 0.05
n_lag = int(round(kim.max_lag / dt)) + 1
lags_index = np.arange(6, 161)  # 0.3-8 s
expected = np.exp(-np.arange(n_lag) * dt) * np.cos(2 * np.pi * np.arange(n_lag) * dt)
expected[0] = 1.0
n_fft = 1 << 16
power = np.ones(n_fft // 2 + 1)
f = np.fft.rfftfreq(n_fft, dt)
for filt in kim.filters:
    power = power * filt.power_response(f)
g = np.fft.irfft(power, n=n_fft)


def g_at(m):
    return g[np.abs(m)]  # symmetric; |m| < n_fft / 2 for every lag used here


rows = np.r_[0, lags_index]  # tau_0 = 0 (normalize: zero_lag) first
n_k = rows.max() + 3000  # g decays well inside 3000 samples (150 s) for kim2023
k_all = np.arange(-n_k, n_k + 1)
whiten = 1 - np.sinc(kim.whitening.width * k_all * dt)
var_e = np.where(k_all == 0, 2.0, 1.0)
# Each e_|k| appears at +k and -k: sum their coefficients before squaring.
coef = g_at(rows[:, None] - k_all[None, :]) * whiten[None, :]
pos = k_all >= 0
folded = coef[:, pos].copy()
folded[:, 1:] += coef[:, k_all < 0][:, ::-1]
cov = folded @ (var_e[pos][:, None] * folded.T)
taper = np.ones(rows.size)  # kim2023: zero_lag_taper = 0
cov = taper[:, None] * cov * taper[None, :]
m = expected[rows] / expected[0]
cov_n = cov - np.outer(m, cov[0]) - np.outer(cov[0], m) + cov[0, 0] * np.outer(m, m)
cov_n = cov_n[1:, 1:]
mine = cov_n / np.sqrt(np.outer(np.diag(cov_n), np.diag(cov_n)))
package = processing_error_correlation(kim, lags_index * dt, dt, expected=expected)
# The package cuts g where |g| < 1e-6 max |g| (assumed there); here it is not cut: 2e-6.
assert np.max(np.abs(mine - package)) < 1e-5, np.max(np.abs(mine - package))

# --- 4. Shrinkage towards a target -------------------------------------------------------------
_write(
    "shrinkage",
    "$$\n\\boxed{\\;\\hat{\\mathbf R} = (1 - \\lambda)\\, \\mathbf R_\\mathrm{sample} + "
    "\\lambda\\, \\mathbf T,\\qquad \\lambda = \\operatorname{clip}\\Bigg(\\frac{\\sum_{i \\ne j} "
    "\\widehat{\\operatorname{Var}}(r_{ij})}{\\sum_{i \\ne j} (r_{ij} - t_{ij})^2},\\ 0,\\ 1\\Bigg)"
    "\\;}\n$$\n\n"
    "$$\n\\widehat{\\operatorname{Var}}(r_{ij}) = \\frac{n}{(n-1)^3} \\sum_{e=1}^{n} "
    "(w_{eij} - \\bar w_{ij})^2,\\qquad w_{eij} = x_{ei}\\, x_{ej}\\quad "
    "(x: \\text{event ACFs standardized per lag})\n$$\n",
)
n_events, p = 25, 12
x = rng.standard_normal((n_events, p)) @ np.linalg.cholesky(ar[:p, :p]).T
xs = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
w = xs[:, :, None] * xs[:, None, :]
r = w.sum(axis=0) / (n_events - 1)
var = n_events / (n_events - 1) ** 3 * np.sum((w - w.mean(axis=0)) ** 2, axis=0)
off = ~np.eye(p, dtype=bool)
lam = np.clip(var[off].sum() / np.sum((r - np.eye(p))[off] ** 2), 0, 1)
shrunk, lam_package = shrink_correlation(x, np.eye(p))
assert np.isclose(lam, lam_package, rtol=1e-10)  # explicit products vs the package's sums
expected_r = (1 - lam) * r + lam * np.eye(p)
np.fill_diagonal(expected_r, 1.0)
assert np.allclose(shrunk, expected_r, atol=1e-12)

print("likelihood formulas: all checks passed")
