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
  for k in range(2, 30):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=k, orth=0)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))

def test_degree_unbiased():
  from primate.diagonalize import lanczos
  from scipy.linalg import eigvalsh_tridiagonal
  np.random.seed(1234)
  n = 150
  A = symmetric(n)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  ew = np.linalg.eigvalsh(A)
  for k in range(5, 150, 5):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=k, orth=15)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))
    rr = eigvalsh_tridiagonal(alpha, beta)
    assert np.allclose(rr.mean(), ew.mean(), atol=0.30)

def test_toeplitz():
  from scipy.linalg import toeplitz
  from primate.diagonalize import lanczos
  from scipy.linalg import eigvalsh_tridiagonal
  np.random.seed(1234)
  n = 150
  c = np.random.uniform(size=n, low=0, high=1.0)
  A = toeplitz(c)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  ew = np.linalg.eigvalsh(A)
  for k in range(25):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=150, orth=15)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))
    rr = eigvalsh_tridiagonal(alpha, beta)
    assert np.allclose(rr.mean(), ew.mean(), atol=np.ptp(ew)*0.05)

## TODO: go back and re-test fttr on toeplitz
# def test_symmetric():
#   assert np.allclose(ew_sym, np.flip(-ew_sym))



def test_quadrature():
  from primate.diagonalize import lanczos, _lanczos
  from sanity import lanczos_quadrature
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n))
  v0 = np.random.uniform(size=A.shape[1])

  ## Test the near-equivalence of the weights and nodes
  alpha, beta = lanczos(A, v0, deg=n, orth=n)  
  nw_test = _lanczos.quadrature(alpha, np.append(0, beta), n, 0)
  nw_true = lanczos_quadrature(A, v=v0, k=n, orth=n)
  assert np.allclose(nw_true[0], nw_test[:,0], atol=1e-6)
  assert np.allclose(nw_true[1], nw_test[:,1], atol=1e-6)

def test_quadrature_methods():
  from primate.diagonalize import _lanczos
  n = 3
  a, b = [1,1,1], [0,1,1]
  nw_test1 = _lanczos.quadrature(a, b, n, 0)
  nw_test2 = _lanczos.quadrature(a, b, n, 1)
  assert np.allclose(nw_test1, nw_test2)

  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  n = 50 
  A = symmetric(n)
  v0 = np.random.uniform(size=A.shape[1])
  a, b = lanczos(A, v0=v0, deg=n)
  a, b = a, np.append([0], b)
  quad1 = np.sum(_lanczos.quadrature(a, b, n, 0).prod(axis=1))
  quad2 = np.sum(_lanczos.quadrature(a, b, n, 1).prod(axis=1))
  assert np.isclose(quad1, quad2, atol=quad1*0.05)

# def test_quadrature_toeplitz():
#   from primate.diagonalize import lanczos, _lanczos
#   from scipy.linalg import toeplitz
#   np.random.seed(1234)

#   A = toeplitz(np.arange(1,8)).astype(np.float32)
#   v0 = np.random.uniform(size=A.shape[1])
#   n = A.shape[1]
#   a, b = lanczos(A, deg=n)
#   a, b = a, np.append([0], b)
#   quad1 = np.sum(_lanczos.quadrature(a, b, n, 0).prod(axis=1))
#   quad2 = np.sum(_lanczos.quadrature(a, b, n, 1).prod(axis=1))
#   assert np.isclose(quad1, quad2, atol=np.abs(quad1)*0.50)

#   ## ground truth
#   ew, ev = np.linalg.eigh(A)
#   mu_0 = np.sum(np.abs(ew))
#   mu_0 * np.ravel(ev[0,:])**2
#   np.array([orth_poly(ew[0], i, mu_0, a, b)**2 for i in range(n)])
#   _lanczos.quadrature(a, b, n, 0)[:,1]
# np.linalg.norm([orth_poly(ew[0], i, mu_0, a, b) for i in range(n)])
# ## to mimick 
# arr = np.array([orth_poly(ew[0], i, mu_0, a, b)**2 for i in range(n)])

  # n = 50 
  # c = np.random.uniform(size=n, low=0, high=1)
  # A = toeplitz(c)
  # v0 = np.random.uniform(size=A.shape[1])
  # a, b = lanczos(A, deg=n)
  # a, b = a, np.append([0], b)
  # quad1 = np.sum(_lanczos.quadrature(a, b, n, 0).prod(axis=1))
  # quad2 = np.sum(_lanczos.quadrature(a, b, n, 1).prod(axis=1))
  # assert np.isclose(quad1, quad2, atol=quad1*0.50)
