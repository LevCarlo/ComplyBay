import numpy as np
from scipy.optimize import newton, root_scalar

def dispersion_eq(k, omega, H, g=9.79329):
    return g * k * np.tanh(k * H) - omega**2

def dispersion_eq_prime(k, omega, H, g=9.79329):
    return g * (np.tanh(k * H) + k * H * (1 / np.cosh(k * H))**2)

def solve_k_brent(omega, H, g=9.79329):
    omega = np.atleast_1d(omega)
    k = np.zeros_like(omega)

    for i in range(len(omega)):
        f = lambda k_: dispersion_eq(k_, omega[i], H, g)
        sol = root_scalar(f, bracket=[1e-6, 1e2], method='brentq')
        k[i] = sol.root

    return k if len(k) > 1 else k[0]

def solve_k_newton(omega, H, g=9.79329):
    omega = np.atleast_1d(omega)
    k = np.zeros_like(omega)

    for i in range(len(omega)):
        k0 = omega[i] / np.sqrt(g * H)
        try:
            k[i] = newton(
                lambda k_: dispersion_eq(k_, omega[i], H, g),
                x0=k0,
                fprime=lambda k_: dispersion_eq_prime(k_, omega[i], H, g))
        except RuntimeError:
            k[i] = solve_k_brent(omega[i], H, g)

    return k if len(k) > 1 else k[0]

def solve_k_poly(omega, H, g=9.79329):
    """
    Refer: https://github.com/s-g-mo/MDN-Compliance/blob/main/forward_funcs/gravd.py
    """
    omega = np.atleast_1d(omega)
    k = np.zeros_like(omega)
    k_deep = omega**2 / g

    a0 = -27 * omega**2 / g
    a1 = np.zeros_like(omega)
    a2 = 27 * H - (9 * omega**2 * H**2)/g
    a3 = np.zeros_like(omega)
    a4 = np.ones_like(omega) * H**3

    for i in range(len(omega)):
        if omega[i] == 0:
            k[i] = 0
        else:
            coeffs = [a4[i], a3[i], a2[i], a1[i], a0[i]]
            roots = np.roots(coeffs)
            positive_real_roots = roots[(roots > 0) & (np.isreal(roots))].real
            if len(positive_real_roots) > 0:
                k[i] = positive_real_roots[0]
            else:
                k[i] = k_deep[i]

        if k_deep[i] * H > 2.96:
            k[i] = k_deep[i]

    return k if len(k) > 1 else k[0]

def solve_k(omega, H, method='poly', g=9.79329):
    if method == 'poly':
        return solve_k_poly(omega, H, g)
    elif method == 'newton':
        return solve_k_newton(omega, H, g)
    elif method == 'brent':
        return solve_k_brent(omega, H, g)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    f = np.linspace(0.004, 0.05, 100)
    w = 2 * np.pi * f
    H = 1500

    k_newton = solve_k(w, H, method='newton')
    k_brent = solve_k(w, H, method='brent')
    k_poly = solve_k(w, H, method='poly')
    

    # Evaluate similarity between methods using mean absolute and relative differences
    pairs = [
        ("Newton", k_newton, "Brent", k_brent),
        ("Newton", k_newton, "Poly", k_poly),
        ("Brent", k_brent, "Poly", k_poly),
    ]

    print(f"{'Comparison':<20}{'Mean Abs Diff':>18}{'Max Abs Diff':>18}{'Min Abs Diff':>18}"
          f"{'Mean Rel Diff':>20}{'Max Rel Diff':>20}{'Min Rel Diff':>20}")
    print("-" * 114)

    for name1, k1, name2, k2 in pairs:
        abs_diff = np.abs(k1 - k2)
        rel_diff = abs_diff / np.maximum(np.abs(k1), np.abs(k2))
        label = f"{name1} vs {name2}"
        print(f"{label:<20}{np.mean(abs_diff):18.3e}{np.max(abs_diff):18.3e}{np.min(abs_diff):18.3e}"
              f"{np.mean(rel_diff):20.6%}{np.max(rel_diff):20.6%}{np.min(rel_diff):20.6%}")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(f, k_newton, label='Newton')
    # ax.plot(f, k_brent, label='Brent')
    # ax.plot(f, k_poly, label='Poly')
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Wave number (1/m)')
    # ax.set_title('Wave number vs Frequency')
    # ax.legend()
    # plt.show()