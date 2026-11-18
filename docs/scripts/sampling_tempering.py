"""Figure sampling_tempering.png: two modes differing in both parameters, with/without tempering.

The setup of tests/test_samplers.py: flat prior on [-6, 6]^2, likelihood with
modes of weight 0.3 at (-3, -3) and 0.7 at (3, 3), each of standard deviation
0.2. Four chains start in the low-weight mode; adaptive Metropolis
(``update="covariance"``) with a one-parameter global jump every 20 steps,
20 000 steps, 4000 burn-in. Plain run against a geometric ladder of 7
temperatures up to 300 (the barrier of 225 in log L becomes < 1 at the
hottest replica). Known answer: weight 0.7 for the upper mode.

(a) Kept draws of chain 0, both runs, and the two modes. (b) Running
fraction of all four chains' kept draws in the upper mode.
"""

import _style
import numpy as np
from _style import plt

from complybay.inference import MetropolisConfig, geometric_temperatures, run_metropolis


class DiagonalModes:
    def log_prior_and_likelihood(self, theta):
        low = -0.5 * np.sum(((theta + 3) / 0.2) ** 2) + np.log(0.3)
        high = -0.5 * np.sum(((theta - 3) / 0.2) ** 2) + np.log(0.7)
        return 0.0, float(np.logaddexp(low, high))


box = (np.full(2, -6.0), np.full(2, 6.0))
common = dict(burn_in=4000, adapt=True, update="covariance", global_jump_interval=20)
starts = np.full((4, 2), -3.0)
plain = run_metropolis(
    DiagonalModes(), starts, MetropolisConfig(20000, 0.2, box, **common), seed=11
)
ladder = geometric_temperatures(7, 300.0)
tempered = run_metropolis(
    DiagonalModes(),
    starts,
    MetropolisConfig(20000, 0.2, box, temperatures=ladder, **common),
    seed=11,
)
assert np.all(plain.samples[..., 0] < 0)  # never leaves the starting mode
weight = np.mean(tempered.samples[..., 0] > 0)
assert abs(weight - 0.7) < 0.06, weight  # 4 x 1600 draws, a few hundred switches: se ~0.02

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
ax_a.plot(*plain.samples[0].T, ".", ms=2, color=_style.APPROX, label="plain")
ax_a.plot(
    *tempered.samples[0].T,
    ".",
    ms=2,
    color=_style.PACKAGE,
    alpha=0.5,
    label=f"tempered ({len(ladder)} temperatures)",
)
for centre, w in ((-3, 0.3), (3, 0.7)):
    ax_a.add_patch(plt.Circle((centre, centre), 0.4, fill=False, color=_style.TRUTH, lw=0.8))
    ax_a.text(centre + 0.5, centre - 0.6, f"weight {w}", fontsize=8)
ax_a.set_xlim(-5, 5)
ax_a.set_ylim(-5, 5)
ax_a.set_aspect("equal")
ax_a.set_xlabel(r"$\theta_1$")
ax_a.set_ylabel(r"$\theta_2$")
ax_a.legend(loc="lower right", markerscale=4)
_style.panel_label(ax_a, "(a)")

upper = (tempered.samples[..., 0] > 0).mean(axis=0)  # fraction over chains per draw
running = np.cumsum(upper) / np.arange(1, upper.size + 1)
ax_b.plot(np.arange(1, upper.size + 1), running, color=_style.PACKAGE, label="tempered")
ax_b.plot(
    np.arange(1, upper.size + 1),
    (plain.samples[..., 0] > 0).mean(axis=0),
    color=_style.APPROX,
    label="plain",
)
ax_b.axhline(0.7, color=_style.TRUTH, lw=0.8, ls="--", label="exact 0.7")
ax_b.set_ylim(-0.05, 1.05)
ax_b.set_xlabel("kept draw")
ax_b.set_ylabel("running fraction in upper mode")
ax_b.legend(loc="lower right")
_style.panel_label(ax_b, "(b)")
fig.tight_layout()
_style.save(fig, __file__)
swaps = np.array([c.swap_acceptance for c in tempered.chains])
print(
    f"upper-mode weight {weight:.3f}; swap acceptance per pair {swaps.min():.2f}-{swaps.max():.2f}"
)
