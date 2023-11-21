import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array

def test_basic():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  A_sparse = csc_array(A)
  v0 = np.random.uniform(size=A.shape[1])
  (a,b), Q = lanczos(A_sparse, v0=v0, tol=1e-8, orth=n-1, return_basis = True)
  assert np.abs(max(eigh_tridiagonal(a,b, eigvals_only=True)) - max(eigsh(A_sparse)[0])) < 1e-4

def test_lanczos_matvec():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  A_sparse = csc_array(A)
  v0 = np.random.uniform(size=A.shape[1])
  (a,b), Q = lanczos(A_sparse, v0=v0, tol=1e-8, orth=0, return_basis = True) 
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
  y = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))
  z = A_sparse @ v0
  assert np.linalg.norm(y - z) < 1e-4

def test_lanczos_accuracy():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it seems likely!
  tol = np.zeros(30)
  for i in range(30):
    v0 = np.random.uniform(size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, tol=1e-8, orth=n-1)
    ew_true = np.sort(eigsh(A, k=n-1, return_eigenvectors=False))
    ew_test = np.sort(eigh_tridiagonal(alpha, beta, eigvals_only=True))
    tol[i] = np.mean(np.abs(ew_test[1:] - ew_true))
  assert np.all(tol < 1e-5)

def test_lanczos_eigenvalues():
  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  n = 10
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  A_sparse = csc_array(A)
  v0 = np.random.uniform(size=A.shape[1])
  k = 3
  alpha, beta = np.zeros(k+1, dtype=np.float32), np.zeros(k+1, dtype=np.float32)
  Q = np.zeros((n,n), dtype=np.float32, order='F')
  tol = 1e-16
  orth = 3
  _lanczos.quadrature(A, v0, k, tol, orth, alpha, beta, Q)


  _lanczos.quadrature

  assert np.abs(max(eigh_tridiagonal(a,b, eigvals_only=True)) - max(eigsh(A_sparse)[0])) < 1e-4


def test_lanczos_correctness():
  from primate.diagonalize import lanczos
  from sanity import lanczos_paige
  np.random.seed(1234)
  assert True
  ## TODO: recombine eigenvalues that are well-separated as a test matrix
  # n = 20
  # A = np.random.uniform(size=(n, n)).astype(np.float32)
  # A = A @ A.T
  # v = np.random.normal(size=n).astype(np.float32)
  # a1, b1 = lanczos_paige(A, v, k = n, tol = 0.0)
  # a2, b2 = lanczos(A, v, max_steps = n, orth=5, tol = 0.0)
  # ew1 = np.sort(eigh_tridiagonal(a1[:n], b1[1:n], eigvals_only=True))
  # ew2 = np.sort(eigh_tridiagonal(a2, b2, eigvals_only=True))
  # ew = np.sort(np.linalg.eigh(A)[0])
  # np.max(np.abs(ew - ew2))
  # np.max(np.abs(ew - ew2))
  # assert np.allclose(ew1, ew2, atol=0.1)

# def test_benchmark():
#   from primate.diagonalize import lanczos, _lanczos_base
#   from scipy.sparse import random, csc_array
#   A = random(1000, 1000, density=0.010)
#   A = csc_array(A @ A.T, dtype=np.float32)
#   assert (A.nnz / (1000**2)) <= 0.30
  
#   import timeit

#   timeit.timeit(lambda: _lanczos_base(A), number=30)
#   timeit.timeit(lambda: lanczos(A), number=30)
#   a, b = lanczos(A)


