import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array, spdiags
from more_itertools import * 

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

from fttr import FTTR_weights
from tqli import tqli


def test_tqli():
  from primate.random import symmetric
  n = 150
  np.random.seed(1234)
  ew = np.random.uniform(size=n, low=0, high=1)
  A = symmetric(n, pd = False, ew = ew)
  a,b = lanczos(A, orth=n, deg=n)
  subdiag = np.append([0], b)
  tqli(a, subdiag)
  assert np.allclose(np.abs(subdiag), 0.0), "Failed to rotate the subdiagonal to 0."
  assert np.allclose(np.sort(a), np.sort(ew)), "Failed to recover the eigenvalues."

# Computing Gaussian quadrature rules with high relative accuracy
# Laudadio, Teresa, Nicola Mastronardi, and Paul Van Dooren. "Computing Gaussian quadrature rules with high relative accuracy." Numerical Algorithms 92.1 (2023): 767-793.
## FTTR works for low-degree polynomial, but has numerical issues beyond small n 
def test_fttr():
  from primate.random import symmetric
  from primate.diagonalize import lanczos
  n = 15
  np.random.seed(1234)
  ew = np.random.uniform(size=n, low=0, high=1)
  A = symmetric(n, pd = False, ew = ew)
  T = lanczos(A, orth=n, deg=n, sparse_mat=True)
  a = T.diagonal(0)
  b = T.diagonal(1)
  rw, Y = np.linalg.eigh(T.todense())

  ## Forward three-term recurrence relation (fttr)
  w_gauss = (np.ravel(Y[0,:])**2)
  w_fttr = np.zeros(n)
  subdiag = np.append([0], b)
  FTTR_weights(ew, a, subdiag, k=n, weights=w_fttr)
  assert np.allclose(np.sort(w_fttr), np.sort(w_gauss)), "FTTR failed to recovered the quadrature weights."
  
  quad_true = np.sum(rw * np.ravel(np.square(Y[0,:])))
  quad_test = np.sum(ew * w_fttr)
  assert np.isclose(quad_true, quad_test), "FTTR failed to recover the quadrature"

  ## Equivalent
  # from primate.diagonalize import _lanczos
  # np.sum(_lanczos.quadrature(a, subdiag, len(a), 0).prod(axis=1))
