import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_array
from typing import Callable
from primate.diagonalize import lanczos
from scipy.linalg import eigh_tridiagonal
import cython

## Base lanczos algorithm, for establishing a baseline

cpdef lanczos_base(A, v0: np.ndarray = None, k: int = None, tol: float = 1e-8):
  n = A.shape[0]
  k = A.shape[1] if k is None else int(k)
  v0 = np.random.uniform(size=A.shape[1], low=-1.0, high=+1.0) if v0 is None else np.array(v0)
  assert k <= n, "Can perform at most k = n iterations"
  assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."
  alpha, beta = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
  qp, qc = np.zeros(n, dtype=np.float32), (v0 / np.linalg.norm(v0)).astype(np.float32)
  for i in range(k):
    qn = A @ qc - beta[i] * qp
    alpha[i] = np.dot(qn, qc)
    qn -= alpha[i] * qc
    beta[i+1] = np.linalg.norm(qn)
    if np.isclose(beta[i+1], tol):
      break
    qn = qn / beta[i+1]
    qp, qc = qc, qn
  return alpha, beta
