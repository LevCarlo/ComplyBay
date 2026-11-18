"""Formulas on docs/sampling.md, checked against the package with numpy.

Writes LaTeX snippets docs/generated/sampling_*.md, which the page includes.
Each formula is written once here, evaluated with numpy, and asserted against
the package (tolerance and reason next to each):

- the Metropolis acceptance at temperature T, against ``run_chain`` on a
  target with an exactly known acceptance probability;
- the Robbins-Monro adaptation of the step size during burn-in, for
  ``update="one"`` and ``"all"``, against the adapted ``proposal_sigma``;
- the replica-exchange (swap) rule, against the package's swap step, and the
  invariance of the product of tempered targets it rests on;
- the geometric ladder, against ``geometric_temperatures``;
- split R-hat and the effective sample size, against ``split_rhat`` and
  ``effective_sample_size`` (autocovariances by direct sums here, by FFT in
  the package).

Run: python docs/scripts/sampling_formulas.py
"""

from __future__ import annotations

import numpy as np
from _style import GENERATED_DIR

from complybay.inference import (
    LogProbTarget,
    MetropolisConfig,
    effective_sample_size,
    geometric_temperatures,
    run_chain,
    split_rhat,
)
from complybay.inference.samplers import _swap


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    (GENERATED_DIR / f"sampling_{name}.md").write_text(text, encoding="utf-8")


rng = np.random.default_rng(0)

# --- 1. Acceptance ----------
_write(
    "metropolis",
    "$$\n\\boxed{\\;\\ln \\alpha = \\big[\\ln p(\\boldsymbol\\theta') - \\ln p(\\boldsymbol\\theta)"
    "\\big] + \\frac{1}{T}\\big[\\ln L(\\boldsymbol\\theta') - \\ln L(\\boldsymbol\\theta)\\big],"
    "\\qquad \\text{accept if } \\ln u < \\ln \\alpha,\\ u \\sim U(0, 1)\\;}\n$$\n",
)


class Step:
    """theta < 0: log L = 0; theta >= 0: log L = -1. Flat prior on the box."""

    def log_prior_and_likelihood(self, theta):
        return 0.0, (0.0 if theta[0] < 0 else -1.0)


# From -0.5 with a tiny step the chain never crosses 0: every proposal is accepted. From
# +0.5 every proposal stays at log L = -1: also accepted. A jump across 0 with proposals
# that always cross tests alpha = exp(-1 / T) directly.
for temperature in (1.0, 3.0):
    box = (np.array([-1.0]), np.array([1.0]))
    config = MetropolisConfig(40000, 1e-9, box, temperature=temperature, global_jump_interval=1)
    chain = run_chain(Step(), [-0.5], config, seed=1)
    # Global jump every step: theta' ~ U(-1, 1), independent of theta. Stationary mass of
    # theta >= 0 is exp(-1/T) / (1 + exp(-1/T)) of the flat prior: exact answer.
    expected = np.exp(-1 / temperature) / (1 + np.exp(-1 / temperature))
    # 40000 draws, correlated over a few steps: standard error ~0.005.
    assert abs(np.mean(chain.samples >= 0) - expected) < 0.02, (temperature, expected)

# --- 2. Robbins-Monro adaptation during burn-in ----------
_write(
    "adaptation",
    "$$\n\\boxed{\\;\\ln \\sigma_{t+1} = \\ln \\sigma_t + \\gamma_t\\,(a_t - a^*),\\qquad "
    "\\gamma_t = (t + 1)^{-0.6},\\qquad t < t_\\mathrm{burn}\\;}\n$$\n\n"
    "$$\na_t \\in \\{0, 1\\}: \\text{step } t \\text{ accepted};\\quad "
    "a^* = 0.44 \\ (\\texttt{one}),\\ 0.234 \\ (\\texttt{all},\\ \\texttt{covariance});\\quad "
    "\\sigma \\le \\text{bound width}\n$$\n",
)
burn_in = 20
box = (np.array([-1e6, -1e6]), np.array([1e6, 1e6]))
t = np.arange(burn_in)
flat = LogProbTarget(lambda theta: 0.0)  # every proposal is accepted: a_t = 1
# update="all": one common factor, so the sigma ratio (1 : 2) is kept.
config = MetropolisConfig(50, [1.0, 2.0], box, burn_in=burn_in, update="all", adapt=True)
chain = run_chain(flat, [0.0, 0.0], config, seed=2)
factor = np.exp(np.sum((t + 1.0) ** -0.6 * (1 - 0.234)))
assert np.allclose(chain.proposal_sigma, [factor, 2 * factor], rtol=1e-12)  # same arithmetic
# update="one": only the perturbed parameter's sigma adapts; with one parameter it is each step.
one_box = (np.array([-1e6]), np.array([1e6]))
chain = run_chain(
    flat, [0.0], MetropolisConfig(50, 1.0, one_box, burn_in=burn_in, adapt=True), seed=3
)
assert np.isclose(
    chain.proposal_sigma[0], np.exp(np.sum((t + 1.0) ** -0.6 * (1 - 0.44))), rtol=1e-12
)

# --- 2b. Adaptive Metropolis (update="covariance") ----------
_write(
    "am",
    "$$\n\\boxed{\\;\\boldsymbol\\theta' = \\boldsymbol\\theta + \\mathbf L \\boldsymbol\\xi,"
    "\\quad \\boldsymbol\\xi \\sim \\mathcal N(0, \\mathbf I),\\quad \\mathbf L \\mathbf L^"
    "\\mathsf{T} = s^2\\, \\mathbf C + \\operatorname{diag}\\big((10^{-4}(b_j - a_j))^2\\big)\\;}"
    "\n$$\n\n"
    "$$\n\\ln s_{t+1} = \\ln s_t + (t - t_c + 1)^{-0.6}\\,(a_t - 0.234),\\qquad "
    "s_{t_c} = \\frac{2.38}{\\sqrt d}\n$$\n\n"
    "$\\mathbf C$: sample covariance of the chain's states from step $t_c/2$ on; $t_c$ = "
    "`covariance_start`; $d$ parameters; $[a_j, b_j]$ the bounds. Both are updated every "
    "step until the end of burn-in, then frozen.\n",
)


class RidgeWithStates:
    """Correlated Gaussian; the details are the state itself, so the trace records it."""

    detail_names = ("x", "y")
    precision = np.linalg.inv(np.array([[1.0, 0.9 * 0.1], [0.9 * 0.1, 0.01]]))

    def log_prior_and_likelihood(self, theta):
        return 0.0, float(-0.5 * theta @ self.precision @ theta)

    def log_prior_likelihood_and_details(self, theta):
        return 0.0, float(-0.5 * theta @ self.precision @ theta), np.array(theta, dtype=float)


am_box = (np.array([-5.0, -0.5]), np.array([5.0, 0.5]))
am_burn, am_start = 600, 200
config = MetropolisConfig(
    800,
    [0.5, 0.05],
    am_box,
    burn_in=am_burn,
    update="covariance",
    adapt=True,
    covariance_start=am_start,
)
chain = run_chain(RidgeWithStates(), [0.0, 0.0], config, seed=5)
states = chain.trace_details  # thin = 1: the state after every step
# The proposal used from the last burn-in step on was built before that step: states and
# acceptances of steps start // 2 .. burn_in - 2.
covariance = np.cov(states[am_start // 2 : am_burn - 1].T)
steps = np.arange(am_start, am_burn - 1)
log_s = np.log(2.38 / np.sqrt(2)) + np.sum(
    (steps - am_start + 1.0) ** -0.6 * (chain.accepted[steps] - 0.234)
)
floor = np.diag((1e-4 * (am_box[1] - am_box[0])) ** 2)
expected = np.exp(2 * log_s) * covariance + floor
assert np.allclose(chain.proposal_covariance, expected, rtol=1e-10)  # Welford vs np.cov

# --- 3. Swaps between temperatures ----------
_write(
    "swap",
    "$$\n\\boxed{\\;\\ln \\alpha_\\mathrm{swap} = \\Big(\\frac{1}{T_i} - \\frac{1}{T_{i+1}}\\Big)"
    "\\big(\\ln L(\\boldsymbol\\theta_{i+1}) - \\ln L(\\boldsymbol\\theta_i)\\big)\\;}"
    "\\qquad \\text{target of replica } i:\\ p(\\boldsymbol\\theta)\\, "
    "L(\\boldsymbol\\theta)^{1/T_i}\n$$\n",
)
# The rule is the Metropolis ratio of the product of the tempered targets (prior cancels).
for _ in range(100):
    lp_a, lp_b, ll_a, ll_b = rng.normal(0, 5, 4)
    t_i, t_j = 1 + rng.random() * 3, 5 + rng.random() * 10
    before = (lp_a + ll_a / t_i) + (lp_b + ll_b / t_j)  # replica i holds a, j holds b
    after = (lp_b + ll_b / t_i) + (lp_a + ll_a / t_j)
    assert np.isclose(after - before, (1 / t_i - 1 / t_j) * (ll_b - ll_a), atol=1e-12)


class _Replica:
    def __init__(self, temperature, log_likelihood):
        self.temperature, self.current, self.theta = temperature, (0.0, log_likelihood), None


for ll_cold, ll_hot in ((-10.0, -12.0), (-12.0, -10.0)):
    accepted, trials = 0, 20000
    swap_rng = np.random.default_rng(4)
    for _ in range(trials):
        pair = [_Replica(1.0, ll_cold), _Replica(4.0, ll_hot)]
        attempts, accepts = np.zeros(1, int), np.zeros(1, int)
        _swap(pair, 0, swap_rng, attempts, accepts)
        accepted += accepts[0]
    expected = min(1.0, np.exp((1 / 1.0 - 1 / 4.0) * (ll_hot - ll_cold)))
    assert abs(accepted / trials - expected) < 0.01  # binomial standard error < 0.0035

_write(
    "ladder",
    "$$\n\\boxed{\\;T_i = T_\\mathrm{max}^{\\,i/(n-1)},\\quad i = 0, \\dots, n-1\\;}\n$$\n",
)
assert np.allclose(geometric_temperatures(5, 30.0), 30.0 ** (np.arange(5) / 4), rtol=1e-14)

# --- 4. Diagnostics ----------
_write(
    "rhat",
    "$$\n\\boxed{\\;\\hat R = \\sqrt{\\frac{\\frac{n-1}{n} W + \\frac{B}{n}}{W}}\\;},\\qquad "
    "W = \\frac{1}{2M} \\sum_{j} s_j^2,\\quad "
    "\\frac{B}{n} = \\operatorname{Var}_j(\\bar x_j)\n$$\n\n"
    "with each of the $M$ chains cut into two halves $j$ of $n$ draws, $s_j^2$ their variances and "
    "$\\bar x_j$ their means (both with $n - 1$ or $2M - 1$ in the denominator).\n",
)
_write(
    "ess",
    "$$\n\\boxed{\\;\\mathrm{ESS} = \\frac{M N}{\\hat\\tau},\\qquad \\hat\\tau = -1 + 2 "
    "\\sum_{k=0}^{K} P_k,\\qquad P_k = \\hat\\rho_{2k} + \\hat\\rho_{2k+1}\\;}\n$$\n\n"
    "$$\n\\hat\\rho_t = 1 - \\frac{W - \\overline{C_t}}{\\widehat{\\operatorname{var}}^+},\\quad "
    "\\widehat{\\operatorname{var}}^+ = \\frac{N-1}{N} W + \\operatorname{Var}_m(\\bar x_m)\n$$\n\n"
    "$C_t$: autocovariance at lag $t$ of each of the $M$ chains of $N$ draws (divided by $N$), "
    "averaged over chains; $W$: within-chain variance; the sum stops before the first "
    "$P_k \\le 0$ and each $P_k$ is capped at the previous one (Geyer's initial monotone "
    "sequence).\n",
)


def rhat_reference(x):
    half = x.shape[1] // 2
    halves = np.concatenate([x[:, :half], x[:, half : 2 * half]])
    n = halves.shape[1]
    w = halves.var(axis=1, ddof=1).mean()
    b_over_n = halves.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * w + b_over_n) / w)


def ess_reference(x):
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    acov = np.array([[np.dot(c[: n - t], c[t:]) / n for t in range(n)] for c in centered])
    w = acov[:, 0].mean() * n / (n - 1)
    var_plus = w * (n - 1) / n + x.mean(axis=1).var(ddof=1)
    rho = 1 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    total, previous = 0.0, np.inf
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        pair = min(pair, previous)
        total, previous = total + pair, pair
    tau = max(-1 + 2 * total, 1 / np.log10(m * n))
    return m * n / tau


ar = np.zeros((4, 800))
noise = rng.standard_normal(ar.shape)
for i in range(1, ar.shape[1]):
    ar[:, i] = 0.8 * ar[:, i - 1] + noise[:, i]
ar[1] += 0.3  # one offset chain
assert np.isclose(split_rhat(ar), rhat_reference(ar), rtol=1e-12)
# FFT vs direct sums of the same estimator: round-off only.
assert np.isclose(effective_sample_size(ar), ess_reference(ar), rtol=1e-9)

print("sampling formulas: all checks passed")
