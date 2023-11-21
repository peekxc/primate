import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_array

## Base lanczos algorithm, for establishing a baseline
def lanczos_base(A, v0: np.ndarray = None, k: int = None, tol: float = 1e-8):
  k = A.shape[1] if k is None else int(k)
  assert k <= A.shape[0], "Can perform at most k = n iterations"
  v0 = np.random.uniform(size=A.shape[1], low=-1.0, high=+1.0) if v0 is None else np.array(v0)
  assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."
  n = A.shape[0]
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

## Paige's A1 variant without additional re-orthogonalization
def lanczos_paige(A, v: np.ndarray, k: int, tol: float = 1e-8):
  assert k <= A.shape[0], "Can perform at most k = n iterations"
  n = A.shape[0]
  alpha, beta = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
  V = np.zeros(shape=(n, 2), dtype=np.float32)
  V[:,0] = v / np.linalg.norm(v)  # v
  V[:,1] = A @ V[:,0]             # u
  for j in range(k):
    alpha[j] = np.dot(V[:,0], V[:,1])
    w = V[:,1] - alpha[j]*V[:,0]
    beta[j+1] = np.linalg.norm(w)
    if np.isclose(beta[j+1], tol):
      break
    vn = w / beta[j+1]
    V[:,1] = A @ vn - beta[j+1] * V[:,0]
    V[:,0] = vn
  return alpha, beta