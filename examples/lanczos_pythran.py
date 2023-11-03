#pythran export lanczos_action(float[], float[:,:] order(C), float[], float[], float[])

import numpy as np 
def lanczos_action(z: np.ndarray, A: np.ndarray, alpha: np.ndarray, beta: np.ndarray, v: np.ndarray):
  """Yields the matrix y = Qz where T(alpha, beta) = Q^T A Q is the tridiagonal matrix spanning K(A, v)"""
  n, k = A.shape[0], len(z)
  assert len(v) == n, "v dimension mismatch"
  y = np.zeros(n) # output vector
  av, bv = np.append([0], alpha), np.append([0,0], beta) 
  qp, qc, qn = np.zeros(n), np.copy(v), np.zeros(n) # previous, current, next
  qc /= np.linalg.norm(v)
  for i in range(1,k+1):
    qn = A @ qc - bv[i]*qp - av[i]*qc 
    y += z[i-1] * qc 
    qp, qc = qc, qn
  return y