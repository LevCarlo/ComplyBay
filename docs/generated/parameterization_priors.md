| `type` | $\ln p_j(\theta_j)$ on $[a, b]$ (`bounds`) | keys |
|---|---|---|
| `uniform` (default) | $0$ | `bounds` |
| `normal` | $-\tfrac12 \big((\theta_j - \mu)/\sigma\big)^2$ | `mean`, `sigma`, `bounds` |
| `half_normal` | $-\tfrac12 \big((\theta_j - a)/s\big)^2$, peak at the lower bound | `scale`, `bounds` |
