import numpy as np 
from primate.diagonalize import lanczos
from scipy.linalg import eigh_tridiagonal

# %% 
np.random.seed(1234)
# ew = np.random.uniform(size=15, low=0, high=2)
ew = 0.2 + 1.5*np.linspace(0, 5, 15)
Q,R = np.linalg.qr(np.random.uniform(size=(15,15)))
A = Q @ np.diag(ew) @ Q.T
A = (A + A.T) / 2
assert np.all(A == A.T)
assert np.allclose(np.linalg.eigvalsh(A) - ew, 0.0)

from primate.random import rademacher
np.random.seed(1234)
# v0 = np.array(np.random.uniform(size=A.shape[0]))
v0 = rademacher(A.shape[1])
(a,b), Q = lanczos(A, v0, max_steps=A.shape[1], orth=0, return_basis=True)

assert np.linalg.norm(Q[:,0] - (v0/np.linalg.norm(v0))) <= 1e-6

rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
y = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))

## We did it! 
assert np.linalg.norm((A @ v0) - y) <= 1e-5


## Now try a matrix function -- it works!
from primate.random import rademacher
np.random.seed(1234)
v0 = rademacher(A.shape[1])
(a,b), Q = lanczos(A, v0, max_steps=A.shape[1], orth=0, return_basis=True)
rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
rw = rw / (rw + 1e-4)
y_test = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))
ew, ev = np.linalg.eig(A)
y_true = (ev @ np.diag((ew / (ew + 1e-4))) @ ev.T) @ v0
assert np.linalg.norm(y_test - y_true) < 1e-3

# %% 
def lanczos_action(z: np.ndarray, A, alpha, beta, v: np.ndarray):
  """Given an input vector 'z', yields the vector y = Qz where T(alpha, beta) = Q^T A Q is the tridiagonal matrix spanning K(A, v)"""
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
  a, b = lanczos(A, v, max_steps=k)   # diagonal and subdiagonal entries of T 
  rr, V = eigh_tridiagonal(a,b)       # Rayleigh-Ritz values + eigenvectors V of T
  z = V @ (f(rr) * V[0,:])            # Compute < f(T), e_1 >
  y = lanczos_action(z, A, a, b, v)
  return np.linalg.norm(v) * y


Q[:,0]
v

## Look at the errors of the action f(A)x as a function of k 
normalize = lambda x: x / np.linalg.norm(x)
identity = lambda x: x
truth_Av = A @ v0
np.linalg.norm(y - truth_Av)

## Not even close!
np.dot(normalize(y), normalize(truth_Av))

# errors = [np.linalg.norm(matrix_function_lanczos(A, identity, v0, j) - truth_Av)  for j in range(1, A.shape[1]+1)]

np.linalg.norm(y - A @ v0)

  # from primate.diagonalize import lanczos
  # np.random.seed(1234)
  # n = 10
  # A = np.random.uniform(size=(n, n)).astype(np.float32)
  # A = A @ A.T
  # A_sparse = csc_array(A)
  # v0 = np.random.uniform(size=A.shape[1])
  # (a,b), Q = lanczos(A_sparse, v0=v0, tol=1e-8, orth=n, return_basis = True) 
  
  
  # e = np.zeros(n)
  # e[0] = 1
  # np.linalg.norm(v0) * (Q @ (V @ np.diag(rw) @ V.T) @ e)
z = A_sparse @ v0
np.linalg.norm(y - z)

# %% 
from bokeh.plotting import figure, show
p = figure(width=200, height=200)








# %% 
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












# %% 



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
