import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_array
from typing import Callable
from primate.diagonalize import lanczos
from scipy.linalg import eigh_tridiagonal

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

## Paige's A(2,7) variant without additional re-orthogonalization
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

def lanczos_quadrature(A, v: np.ndarray, **kwargs):
  """Lanczos quadrature.
  
  Computes the quadrature nodes { n_i }_{i=1}^k and weights { t_i }_{i=1}^k representing the Gaussian quadrature rule

  Based on Algorithm 2.1 of "An analysis on stochastic Lanczos quadrature with asymmetric quadrature nodes"
  """
  k = kwargs.pop("k", A.shape[1])
  (a,b) = lanczos(A, v, deg = k, return_basis = False, **kwargs)
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
  return rw, (V[0,:]**2)

def girard_hutch(A, f: Callable, nv: int = 150, estimates: bool = False, **kwargs):
  """Girard-Hutchinson estimator"""
  tr_est = 0.0
  est = np.zeros(nv)
  for i in range(nv):
    v0 = np.random.choice([-1, +1], size=A.shape[1])
    c = np.linalg.norm(v0)**2
    theta, tau = lanczos_quadrature(A, v0, **kwargs)
    est[i] = c * np.sum(f(theta) * tau)
  return np.sum(est) / nv if not estimates else est

def approx_matvec(A, v: np.ndarray, k: int = None, f: Callable = None):
  k = A.shape[1] if k is None else int(k)
  (a,b), Q = lanczos(A, v, deg=k, orth=0, return_basis=True)
  Q = Q[:,:k]
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)  # lanczos_quadrature(A, v, )
  rw = rw if f is None else f(rw)
  y = np.linalg.norm(v) * (Q @ V @ (V[0,:] * rw))
  return y

def orthogonal_polynomial_value(x, k, theta, gamma):
  # Initialize the polynomials p_{-1} and p_0
  p_minus1 = 0
  p_0 = 1.0 / np.sqrt(1)  # Since k_0 = 1 / sqrt(mu_0) and mu_0 = 1

  # Use recurrence relation to compute p_k for k > 0
  p_k = 0.0
  for ell in range(1, k + 1):
    p_k = ((x - theta[ell - 1]) * p_0 - gamma[ell - 1] * p_minus1) / gamma[ell]

    # Update values for next iteration
    p_minus1 = p_0
    p_0 = p_k
  return p_k

from scipy.sparse import spdiags
alpha = np.array([1,1,1])
beta = np.array([1,1,0])
T = spdiags(data=[beta, alpha, np.roll(beta, 1)], diags=(-1, 0, +1), m=3, n=3).todense()
ew, ev = np.linalg.eigh(T)

a = alpha 
b = np.array([0,1,1])
p0 = lambda x: 1 / np.sqrt(np.sum(np.abs(ew)))
p1 = lambda x: ((x - alpha[0]) * p0(x)) / beta[1]
# p2 = lambda x: ((x - alpha[1]) * p0(x)) / beta[1]


ev[0,:]
1 / np.sqrt(np.sum(np.abs(ew)))
orthogonal_polynomial_value(0.5, 1, alpha, beta[:2])
