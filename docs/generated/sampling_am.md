$$
\boxed{\;\boldsymbol\theta' = \boldsymbol\theta + \mathbf L \boldsymbol\xi,\quad \boldsymbol\xi \sim \mathcal N(0, \mathbf I),\quad \mathbf L \mathbf L^\mathsf{T} = s^2\, \mathbf C + \operatorname{diag}\big((10^{-4}(b_j - a_j))^2\big)\;}
$$

$$
\ln s_{t+1} = \ln s_t + (t - t_c + 1)^{-0.6}\,(a_t - 0.234),\qquad s_{t_c} = \frac{2.38}{\sqrt d}
$$

$\mathbf C$: sample covariance of the chain's states from step $t_c/2$ on; $t_c$ = `covariance_start`; $d$ parameters; $[a_j, b_j]$ the bounds. Both are updated every step until the end of burn-in, then frozen.
