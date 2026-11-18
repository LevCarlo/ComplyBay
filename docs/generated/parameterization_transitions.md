$$
\text{sigmoid:}\quad S(z) = \frac{L(k u) - L(-k h)}{L(k h) - L(-k h)},\quad u = \operatorname{clip}(z - z_i, -h, h),\quad L(x) = \frac{1}{1 + e^{-x}},\quad \boxed{\,k = \frac{2 \ln 9}{w},\ h = 2w\,}
$$

$$
\text{smootherstep:}\quad S(z) = 6x^5 - 15x^4 + 10x^3,\quad x = \operatorname{clip}\Big(\frac{z - z_i + h}{2h}, 0, 1\Big),\quad \boxed{\,h = \frac{w}{2 \times 0.50673} = 0.9867\,w\,}
$$

| shape | half-support $h$ | $S$ at $z_i \mp w/2$ | steepest slope $\max\, dS/dz$ | smoothness |
|---|---|---|---|---|
| `sigmoid` | $2w$ (logistic tail there: 1.5e-04) | 0.1, 0.9 (to $3\times10^{-5}$) | $1.099/w$ | slope jumps by $6.7e-04/w$ at $\pm h$ |
| `smootherstep` | $0.9867\,w$ | 0.1, 0.9 | $0.950/w$ | $C^2$ |
