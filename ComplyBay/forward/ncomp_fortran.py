'''
FUNCTION ncomp_fortran.py

Translated from Wayne Crawford's MATLAB code for forward calculating normalized
compliance from synthetic Earth structure  

Original source can be found at (http://www.ipgp.fr/~crawford/Homepage/Software.html).

Stephen Mosher, Feb 2020. 
'''

#################################### IMPORTS ###################################

# The usual.
import numpy as np
import time

# Import helper functions - raydep_ft does the heavy lifting. It's an F95
# file and it has to be compiled using F2PY from NumPy.
# from forward_funcs import gravd, 
# start = time.time()
from ComplyBay.forward import raydep_ft, k_solver
# print("import time:", time.time() - start)
################################### FUNCTION ###################################

def ncomp_fortran(depth, freqs, model, log_scale=False):
  # Compute wavenumber of infragravity waves and slowness vectors.
  ω = 2 * np.pi * freqs
  # k = gravd.gravd(ω, depth)
  k = k_solver.solve_k(ω, depth, method='poly')
  p = k / ω

  # Compute normalized compliance.
  ncomp = np.zeros(len(ω))
  for i in range(len(p)):
    v, u, sigzz, sigzx = raydep_ft.raydep_ft(p[i], ω[i], model)
    ncomp[i] = -k[i] * v[0] / (ω[i] * sigzz[0])
  # return ncomp
  return np.log10(ncomp) if log_scale else ncomp


# Vectorized batch version using raydep_batch_ft
def ncomp_fortran_batch(depth, freqs, model, log_scale=False):
    """
    Vectorized version of ncomp_fortran using the raydep_batch_ft Fortran interface.
    """
    ω = 2 * np.pi * freqs
    # start = time.time()
    k = k_solver.solve_k(ω, depth, method='poly')
    # print("k solver:", time.time() - start)
    p = k / ω

    # Call Fortran batch routine
    # start = time.time()
    v_all, u_all, zz_all, zx_all = raydep_ft.raydep_batch_ft(p, ω, model, model.shape[0], len(p))
    # print("raydep_batch_ft:", time.time() - start)

    # Compute normalized compliance from Fortran outputs
    ncomp = -k * v_all[0, :] / (ω * zz_all[0, :])
    return np.log10(ncomp) if log_scale else ncomp

if __name__ == "__main__":
  inpModel = np.array([[1000, 2.500, 4.000, 2.500],
                       [2000, 3.000, 7.000, 4.000],
                         [400, 2.500, 4.000, 1.000],
                         [1000, 3.000, 7.000, 4.000]])
  
  freqs = np.arange(0.003, 0.02 + 0.001, 0.001)
  wdepth = 3000
  # ncomp = ncomp_fortran(wdepth, freqs, inpModel)
  ncomp = ncomp_fortran_batch(wdepth, freqs, inpModel)
  print(ncomp)
  