$$
\boxed{\;\mathrm{ESS} = \frac{M N}{\hat\tau},\qquad \hat\tau = -1 + 2 \sum_{k=0}^{K} P_k,\qquad P_k = \hat\rho_{2k} + \hat\rho_{2k+1}\;}
$$

$$
\hat\rho_t = 1 - \frac{W - \overline{C_t}}{\widehat{\operatorname{var}}^+},\quad \widehat{\operatorname{var}}^+ = \frac{N-1}{N} W + \operatorname{Var}_m(\bar x_m)
$$

$C_t$: autocovariance at lag $t$ of each of the $M$ chains of $N$ draws (divided by $N$), averaged over chains; $W$: within-chain variance; the sum stops before the first $P_k \le 0$ and each $P_k$ is capped at the previous one (Geyer's initial monotone sequence).
