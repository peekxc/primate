import numpy as np 
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')


# from importlib_resources import files
# files('primate')

def challenge1():
  pass 
  from scipy.sparse import load_npz
  data_dir = primate.__file__[:(primate.__file__.find('primate') + 8)] + "data"
  A = load_npz(data_dir + '/challenge_1328_rank871.npz')
  # largest: 5.857450021586329
  # smallest: 0.0038922391321746437
  top_ew = np.take(eigsh(A, k=1, which='LM', return_eigenvectors=False), 0)
  low_ew = eigsh(A, which='SM', ncv = A.shape[0], k=A.shape[0] - 872, return_eigenvectors=False)
  gap = min(low_ew[~np.isclose(low_ew, 0.0)])

  ## This seems to work 
  from primate.functional import estimate_spectral_radius
  from primate.diagonalize import lanczos
  ## => means lanczos up to degree 'deg' should yield rayleigh ritz value w/ rel error <= 0.01
  ## This uses the theory to do so!
  eps = 0.1
  deg = estimate_spectral_radius(A, rtol=eps)
  a, b = lanczos(A, deg=deg)
  rr_max = eigvalsh_tridiagonal(a,b, select='i', select_range=[deg-1, deg-1])[0]
  assert np.abs((rr_max - top_ew) / top_ew) <= eps
  assert top_ew <= (rr_max / (1.0 - eps))


  ## Estimating the spectral grap is quite hard!
  from primate.trace import hutch
  s0 = np.array([hutch(A, fun="smoothstep", a=0, b=gap * 2, maxiter=30, seed=i) for i in range(250)])
  s1 = np.array([hutch(A, fun="smoothstep", a=0, b=gap, maxiter=30, seed=i) for i in range(250)])
  s2 = np.array([hutch(A, fun="smoothstep", a=0, b=gap / 2, maxiter=30, seed=i) for i in range(250)])
  s3 = np.array([hutch(A, fun="smoothstep", a=0, b=gap / 4, maxiter=30, seed=i) for i in range(250)])
  s4 = np.array([hutch(A, fun="smoothstep", a=0, b=gap / 8, maxiter=30, seed=i) for i in range(250)])
  s5 = np.array([hutch(A, fun="smoothstep", a=0, b=gap / 16, maxiter=30, seed=i) for i in range(250)])

  ## Indeed, whereas error is relatively small (~3) for gap * 2 and gap, the computation is highly unstable 
  ## if the gap is set too low! On the other hand, the computation introduces bias if gap is too large!
  np.mean(np.abs(s0 - 871))
  np.mean(np.abs(s1 - 871))
  np.mean(np.abs(s2 - 871))
  np.mean(np.abs(s3 - 871))
  np.mean(np.abs(s4 - 871))
  np.mean(np.abs(s5 - 871))

  s0 = np.array([hutch(A, fun="smoothstep", a=gap, b=gap * 2, maxiter=30, seed=i) for i in range(250)])
  s1 = np.array([hutch(A, fun="smoothstep", a=gap / 2, b=gap, maxiter=30, seed=i) for i in range(250)])
  s2 = np.array([hutch(A, fun="smoothstep", a=gap / 4, b=gap / 2, maxiter=30, seed=i) for i in range(250)])
  s3 = np.array([hutch(A, fun="smoothstep", a=gap / 8, b=gap / 4, maxiter=30, seed=i) for i in range(250)])
  s4 = np.array([hutch(A, fun="smoothstep", a=gap / 16, b=gap / 8, maxiter=30, seed=i) for i in range(250)])
  s5 = np.array([hutch(A, fun="smoothstep", a=gap / 32, b=gap / 16, maxiter=30, seed=i) for i in range(250)])

  ## Strange enough, bringing up the gap introduces a lot of error, which ends up decreasing
  ## as s0 -> s5. This could just be because s5 has larger total distance |b - a| 
  np.mean(np.abs(s0 - 871))
  np.mean(np.abs(s1 - 871))
  np.mean(np.abs(s2 - 871))
  np.mean(np.abs(s3 - 871))
  np.mean(np.abs(s4 - 871))
  np.mean(np.abs(s5 - 871))

  ## In contrast, the error remains stable if we remain above the gap 
  ## So, ideally, we want to use an upper-bound on the gap
  s0 = np.array([hutch(A, fun="smoothstep", a=0, b=gap * 10, maxiter=30, seed=i) for i in range(250)]) 
  np.mean(np.abs(s0 - 871))

  ## 
  s0 = np.array([hutch(A, fun="smoothstep", a=0, b=gap * 10, maxiter=30, seed=i, deg=60, orth=10) for i in range(250)]) 
  np.mean(np.abs(s0 - 871))
