import numpy as np 

# %% 
np.random.seed(1234)
ew = np.random.uniform(size=15, low=0, high=2)
Q,R = np.linalg.qr(np.random.uniform(size=(15,15)))
A = Q @ np.diag(ew) @ Q.T
A = (A + A.T) / 2
assert np.all(A == A.T)
# from numpy.polynomial.polynomial import polyfit
# polyfit()


# %% 
from primate.diagonalize import lanczos
from scipy.linalg import eigh_tridiagonal
np.random.seed(1234)
v0 = np.array(np.random.uniform(size=A.shape[0]))
a,b = lanczos(A, v0, max_steps=10)

def lanczos_action(z: np.ndarray, A, alpha, beta, v: np.ndarray):
  """Yields the matrix y = Qz where T(alpha, beta) = Q^T A Q is the tridiagonal matrix spanning K(A, v)"""
  n, k = A.shape[0], len(z)
  assert len(v) == n, "v dimension mismatch"
  y = np.zeros(n) # output vector
  av, bv = np.append([0], alpha), np.append([0,0], beta) 
  qp, qc = np.zeros(n), v / np.linalg.norm(v) # previous, current
  for i in range(1,k+1):
    qn = A @ qc - bv[i]*qp - av[i]*qc 
    y += z[i-1]*qc 
    qp, qc = qc, qn 
  return y

from typing import *
def matrix_function_lanczos(A, f: Callable, v: np.ndarray, k: int = 15):
  """Approximates the action v -> f(A)v via the Lanczos method"""
  assert len(v) == A.shape[0], "Dimension mismatch"
  k = min(k, A.shape[0])
  a, b = lanczos(A, v, max_steps=k)  # diagonal and subdiagonal entries of T 
  rr, V = eigh_tridiagonal(a,b[:-1]) # Rayleigh-Ritz values + eigenvectors V of T
  z = V @ (f(rr) * V[0,:])           # Compute < f(T), e_1 >
  y = lanczos_action(z, A, a, b, v)
  return np.linalg.norm(v) * y

identity = lambda x: x
truth_Av = A @ v0

matrix_function_lanczos(A, identity, v0, A.shape[0])

normalize = lambda x: x / np.linalg.norm(x)


from primate.random import rademacher
samples, samples_mean = [], []
for i in range(300):
  rv = rademacher(A.shape[0])
  rv_f = matrix_function_lanczos(A, identity, rv, A.shape[0])
  samples.append(rv_f)
  samples_mean.append(np.dot(rv, rv_f))

mean_trace = np.cumsum(samples_mean)/np.arange(1,301)

np.dot(normalize(A @ v0), normalize(v0)) # 

## Truth 
a,b = lanczos(A, v0, max_steps=A.shape[0], sparse_mat=False)
T = lanczos(A, v0, max_steps=A.shape[0], sparse_mat=True)
y = np.linalg.norm(v0) * lanczos_action(T @ np.array([1] + [0]*(T.shape[1]-1)), A, a, b, v0)

np.dot(normalize(y), normalize(A @ v0))

## Can do block-wise computation
eigh_tridiagonal(a,b[:-1], select='i', select_range=(0,0))







## XTRACE on (n x n) matrix A
import numpy as np 
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse import spdiags, diags
# spsolve_triangular(R, diags(np.ones(m)))

np.trace(A) ## the goal

xtrace(A, 30, method='rademacher')

xtrace(A, 10, method='sphere')
xtrace(A, 10, method='improved')

## NOTE: Xtrace doesn't do this; see xtrace_tol
# samples_xtrace = np.ravel([xtrace(A, 30, method='rademacher')[0] for i in range(100)])
# samples_xtrace = np.cumsum(samples_xtrace) / np.arange(1, len(samples_xtrace)+1)
show(figure_trace(samples_xtrace))

tr_est, info = slq(A, max_num_samples=100, return_info=True)

show(figure_trace(np.ravel(info['convergence']['samples'])))




# %% 
# def lanczos(A, v, k: int):
#   v /= np.linalg.norm(v)
#   u = A @ v
#   alpha, beta = np.zeros(A.shape[0]), np.zeros(A.shape[0])
#   for i in range(k):
#     alpha[i] = np.dot(v, u)
#     w = u - alpha[i] * v
#     beta[i+1] = np.linalg.norm(w)
#     vn = w / beta[i+1]
#     u
