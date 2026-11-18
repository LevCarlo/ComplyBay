$$
\boxed{\;\hat{\mathbf R} = (1 - \lambda)\, \mathbf R_\mathrm{sample} + \lambda\, \mathbf T,\qquad \lambda = \operatorname{clip}\Bigg(\frac{\sum_{i \ne j} \widehat{\operatorname{Var}}(r_{ij})}{\sum_{i \ne j} (r_{ij} - t_{ij})^2},\ 0,\ 1\Bigg)\;}
$$

$$
\widehat{\operatorname{Var}}(r_{ij}) = \frac{n}{(n-1)^3} \sum_{e=1}^{n} (w_{eij} - \bar w_{ij})^2,\qquad w_{eij} = x_{ei}\, x_{ej}\quad (x: \text{event ACFs standardized per lag})
$$
