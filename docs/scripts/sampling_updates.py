"""Figure sampling_updates.png: the three updates on a correlated Gaussian ridge.

The setup of tests/test_samplers.py: a 2-D Gaussian with correlation 0.99
between parameters of standard deviation 1 and 0.01, proposals started at
(0.5, 0.005), 30 000 steps, 10 000 of them burn-in with adaptation, one
chain per update, same seed. Known answer: the target covariance (asserted
for ``covariance``) and an effective sample size that grows from ``one`` to
``all`` to ``covariance``.

(a) The first 300 kept draws of each update over the target's 1- and
2-sigma ellipses. (b) Autocorrelation of the first parameter against the lag
in steps, with the effective sample size of the 20 000 kept draws.
"""

import _style
import numpy as np
from _style import plt

from complybay.inference import LogProbTarget, MetropolisConfig, effective_sample_size, run_chain

STD = np.array([1.0, 0.01])
CORR = 0.99
COV = np.array([[STD[0] ** 2, CORR * STD[0] * STD[1]], [CORR * STD[0] * STD[1], STD[1] ** 2]])
PRECISION = np.linalg.inv(COV)
BOX = (np.array([-10.0, -0.1]), np.array([10.0, 0.1]))


def ridge(theta):
    return float(-0.5 * theta @ PRECISION @ theta)


chains, ess = {}, {}
for update in ("one", "all", "covariance"):
    config = MetropolisConfig(30000, [0.5, 0.005], BOX, burn_in=10000, update=update, adapt=True)
    chains[update] = run_chain(LogProbTarget(ridge), [0.0, 0.0], config, seed=0)
    ess[update] = float(effective_sample_size(chains[update].samples[None, :, 0]))
assert ess["one"] < ess["all"] < ess["covariance"], ess
np.testing.assert_allclose(np.cov(chains["covariance"].samples.T), COV, rtol=0.15, atol=1e-6)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
angle = np.linspace(0, 2 * np.pi, 200)
chol = np.linalg.cholesky(COV)
for radius in (1, 2):
    ring = radius * chol @ np.vstack([np.cos(angle), np.sin(angle)])
    ax_a.plot(ring[0], ring[1] * 100, color=_style.TRUTH, lw=0.8)
colors = {"one": _style.APPROX, "all": _style.SECOND, "covariance": _style.PACKAGE}
for update, chain in chains.items():
    first = chain.samples[:300]
    ax_a.plot(
        first[:, 0], first[:, 1] * 100, "-", lw=0.6, color=colors[update], alpha=0.8, label=update
    )
    x = chain.samples[:, 0] - chain.samples[:, 0].mean()
    lags = np.arange(0, 400)
    acf = np.array([np.dot(x[: x.size - k], x[k:]) for k in lags]) / np.dot(x, x)
    ax_b.plot(lags, acf, color=colors[update], lw=1.3, label=f"{update}: ESS {ess[update]:.0f}")
ax_a.set_xlabel(r"$\theta_1$")
ax_a.set_ylabel(r"$100\,\theta_2$")
ax_a.legend(loc="lower right")
_style.panel_label(ax_a, "(a)")
ax_b.axhline(0, color=_style.TRUTH, lw=0.6)
ax_b.set_xlabel("lag (steps)")
ax_b.set_ylabel(r"autocorrelation of $\theta_1$")
ax_b.legend(loc="upper right", title="20 000 kept draws")
_style.panel_label(ax_b, "(b)")
fig.tight_layout()
_style.save(fig, __file__)
print({k: round(v) for k, v in ess.items()})
print({k: round(c.acceptance_rate(after_burn_in=10000), 3) for k, c in chains.items()})
