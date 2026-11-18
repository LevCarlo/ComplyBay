| `method` | relation (km/s, g/cm³) | stated range | source |
|---|---|---|---|
| Vp `brocher2005` | $V_p = 0.9409 + 2.0947V_s - 0.8206V_s^2 + 0.2683V_s^3 - 0.0251V_s^4$ | $0 \le V_s \le 4.5$ (enforced) | Brocher (2005), Eq. 9 |
| Vp `castagna1985` | $V_p = 1.16\,V_s + 1.36$ | none enforced | Castagna et al. (1985), mudrock line |
| Vp `constant_vp_vs` | $V_p = r\,V_s$, $r$ = `vp_vs` | none | - |
| ρ `brocher2005` | $\rho = 1.6612V_p - 0.4721V_p^2 + 0.0671V_p^3 - 0.0043V_p^4 + 0.000106V_p^5$ | $1.5 \le V_p \le 8.5$ (enforced) | Brocher (2005), Eq. 1 (fit to Nafe-Drake) |
| ρ `gardner1974` | $\rho = a\,V_p^{\,b}$, $a = 1.741$, $b = 0.25$ (options `a`, `b`) | none | Gardner et al. (1974) |
| ρ `constant` | $\rho$ = `rho` | none | - |
