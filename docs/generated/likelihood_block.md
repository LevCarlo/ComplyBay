$$
\mathbf z = \left(\frac{d_i - g_i}{s_i}\right)_i,\quad \mathbf R = \mathbf V \boldsymbol\Lambda \mathbf V^\mathsf{T},\quad \mathbf Q = \mathbf V_{k} \boldsymbol\Lambda_{k}^{-1/2},\quad k = \min\Big\{k : \sum_{i \le k} \lambda_i \ge f \sum_i \lambda_i\Big\}
$$

$$
\boxed{\;\ln L_\mathrm{block} = -\frac12\, \mathbf y^\mathsf{T} \mathbf M^{-1} \mathbf y - \frac12 \ln\det \mathbf M,\qquad \mathbf y = \mathbf Q^\mathsf{T} \mathbf z,\quad \mathbf M = \mathbf Q^\mathsf{T} \mathbf D \mathbf R \mathbf D \mathbf Q\;}
$$

$$
\mathbf D = \operatorname{diag}(h \text{ of the member owning each sample});\qquad \text{one } h:\quad \ln L_\mathrm{block} = -\frac{\lvert \mathbf y \rvert^2}{2h^2} - k \ln h
$$
