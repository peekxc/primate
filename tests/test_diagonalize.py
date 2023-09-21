import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator

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