import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

def test_basic():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n))
  v0 = np.random.uniform(size=n)
  (a,b), Q = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=n-1, return_basis = True)
  assert np.abs(max(eigh_tridiagonal(a,b, eigvals_only=True)) - max(eigsh(A_sparse)[0])) < 1e-4

def test_matvec():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n))
  v0 = np.random.uniform(size=n)
  (a,b), Q = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=0, return_basis = True) 
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
  y = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))
  z = A_sparse @ v0
  assert np.linalg.norm(y - z) < 1e-4

def test_accuracy():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = symmetric(n).astype(np.float64)
  alpha, beta = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  tol = np.zeros(30)
  for i in range(30):
    v0 = np.random.uniform(size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-8, orth=n-1)
    ew_true = np.sort(eigsh(A, k=n-1, return_eigenvectors=False))
    ew_test = np.sort(eigh_tridiagonal(alpha, beta, eigvals_only=True))
    tol[i] = np.max(np.abs(ew_test[1:] - ew_true))
  assert np.all(tol < 1e-12)

def test_high_degree():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = symmetric(n)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  tol = np.zeros(30)
  for k in range(2, 30):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=k, orth=0)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))

def test_quadrature():
  from primate.diagonalize import lanczos, _lanczos
  from sanity import lanczos_quadrature
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n))
  v0 = np.random.uniform(size=A.shape[1])

  ## Test the near-equivalence of the weights and nodes
  alpha, beta = lanczos(A, v0, deg=n, orth=n)  
  nw_test = _lanczos.quadrature(alpha, np.append(0, beta), n)
  nw_true = lanczos_quadrature(A, v=v0, k=n, orth=n)
  assert np.allclose(nw_true[0], nw_test[:,0], atol=1e-6)
  assert np.allclose(nw_true[1], nw_test[:,1], atol=1e-6)