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

def test_matvec():
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
  
  # e = np.zeros(n)
  # e[0] = 1
  # np.linalg.norm(v0) * (Q @ (V @ np.diag(rw) @ V.T) @ e)
  z = A_sparse @ v0
  np.linalg.norm(y - z)

  assert np.abs(max() - max(eigsh(A_sparse)[0])) < 1e-4


def test_lanczos():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  # lo = aslinearoperator(A)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it seems likely!
  tol = np.zeros(30)
  for i in range(30):
    v0 = np.random.uniform(size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, tol=1e-8, orth=n-1)
    ew = np.sort(eigh_tridiagonal(alpha, beta[:-1], eigvals_only=True))
    tol[i] = np.mean(np.abs(ew[1:] - np.sort(eigsh(A, k=n-1, return_eigenvectors=False))))
  assert np.all(tol < 1e-5)


def test_benchmark():
  from primate.diagonalize import lanczos, _lanczos_base
  from scipy.sparse import random, csc_array
  A = random(1000, 1000, density=0.010)
  A = csc_array(A @ A.T, dtype=np.float32)
  assert (A.nnz / (1000**2)) <= 0.30
  
  import timeit

  timeit.timeit(lambda: _lanczos_base(A), number=30)
  timeit.timeit(lambda: lanczos(A), number=30)
  a, b = lanczos(A)


