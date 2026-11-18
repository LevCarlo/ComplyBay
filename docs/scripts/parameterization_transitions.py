"""Figure parameterization_transitions.png: the two transition shapes of a gradational interface.

(a) Weight S of the lower segment against (z - z_i) / w for ``sigmoid`` and
``smootherstep``, with the 10-90 % width and each half-support h marked.
(b) Slope dS/dz in units of 1 / w. Known answer: S = 0.1, 0.5, 0.9 at
-w/2, 0, w/2 and exactly 0 / 1 outside -/+h (asserted).
"""

import _style
import numpy as np
from _style import plt

from complybay.parameterization import transition_half_support_km, transition_weight

WIDTH = 1.0  # km; the plot is in units of w
x = np.linspace(-2.4, 2.4, 4801)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.0))
for shape, color in (("sigmoid", _style.PACKAGE), ("smootherstep", _style.SECOND)):
    s = transition_weight(x, 0.0, WIDTH, shape)
    half = transition_half_support_km(WIDTH, shape)
    marks = transition_weight(np.array([-0.5, 0.0, 0.5]), 0.0, WIDTH, shape)
    assert np.allclose(marks, [0.1, 0.5, 0.9], atol=1e-3)
    assert np.all(s[x <= -half] == 0) and np.all(s[x >= half] == 1)
    ax_a.plot(x, s, color=color, lw=1.4, label=f"{shape} (h = {half:.3g} w)")
    ax_b.plot(x, np.gradient(s, x), color=color, lw=1.4, label=shape)
    for ax in (ax_a, ax_b):
        for edge in (-half, half):
            ax.axvline(edge, color=color, ls=":", lw=0.9)
ax_a.axvspan(-0.5, 0.5, color=_style.APPROX, alpha=0.15, lw=0, label="10-90 % width w")
ax_a.axhline(0.1, color=_style.APPROX, lw=0.6)
ax_a.axhline(0.9, color=_style.APPROX, lw=0.6)
ax_a.set_xlabel(r"$(z - z_i)\,/\,w$")
ax_a.set_ylabel("weight S of the lower segment")
ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
_style.panel_label(ax_a, "(a)")
ax_b.set_xlabel(r"$(z - z_i)\,/\,w$")
ax_b.set_ylabel(r"$w\,dS/dz$")
ax_b.legend(loc="upper right")
_style.panel_label(ax_b, "(b)")
fig.tight_layout()
_style.save(fig, __file__)
