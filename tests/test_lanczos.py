import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 

## Add the test directory to the sys path 
import sys
import primate
ind = [i for i, c in enumerate(primate.__file__) if c == '/'][-3]
sys.path.insert(0, primate.__file__[:ind] + '/tests')

def gen_sym(n: int, ew: np.ndarray = None):
  ew = 0.2 + 1.5*np.linspace(0, 5, n) if ew is None else ew
  Q,R = np.linalg.qr(np.random.uniform(size=(n,n)))
  A = Q @ np.diag(ew) @ Q.T
  A = (A + A.T) / 2
  return A

def test_basic():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  A_sparse = csc_array(A)
  v0 = np.random.uniform(size=A.shape[1])
  (a,b), Q = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=n-1, return_basis = True)
  assert np.abs(max(eigh_tridiagonal(a,b, eigvals_only=True)) - max(eigsh(A_sparse)[0])) < 1e-4

def test_matvec():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  A_sparse = csc_array(A)
  v0 = np.random.uniform(size=A.shape[1])
  (a,b), Q = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=0, return_basis = True) 
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
  y = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))
  z = A_sparse @ v0
  assert np.linalg.norm(y - z) < 1e-4

def test_accuracy():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  tol = np.zeros(30)
  for i in range(30):
    v0 = np.random.uniform(size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-8, orth=n-1)
    ew_true = np.sort(eigsh(A, k=n-1, return_eigenvectors=False))
    ew_test = np.sort(eigh_tridiagonal(alpha, beta, eigvals_only=True))
    tol[i] = np.mean(np.abs(ew_test[1:] - ew_true))
  assert np.all(tol < 1e-5)

def test_high_degree():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  tol = np.zeros(30)
  for k in range(2, 30):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=k, orth=0)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))

def test_quadrature():
  from primate.diagonalize import lanczos, _lanczos
  from sanity import lanczos_quadrature
  np.random.seed(1234)
  n = 10
  A = np.random.uniform(size=(n, n)).astype(np.float32)
  A = A @ A.T
  A_sparse = csc_array(A)
  v0 = np.random.uniform(size=A.shape[1])

  ## Test the near-equivalence of the weights and nodes
  alpha, beta = lanczos(A_sparse, v0, deg=n, orth=n)  
  nw_test = _lanczos.quadrature(alpha, np.append(0, beta), n)
  nw_true = lanczos_quadrature(A, v=v0, k=n, orth=n)
  assert np.allclose(nw_true[0], nw_test[:,0], atol=1e-6)
  assert np.allclose(nw_true[1], nw_test[:,1], atol=1e-6)


## TODO: see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quadrature.html for alternative tol and rtol 
def test_stochastic_quadrature():
  from primate.diagonalize import _lanczos
  assert hasattr(_lanczos, "stochastic_quadrature"), "Module compile failed"
  
  from primate.trace import sl_gauss
  np.random.seed(1234)
  A = gen_sym(30)
  n = A.shape[1]
  nv, lanczos_deg = 250, 20
  A = csc_array(A, dtype=np.float32)

  ## Test raw computation to ensure it's returning valid entries
  # A, nv, dist, engine_id, seed, lanczos_degree, lanczos_rtol, orth, ncv, num_threads
  quad_nw = _lanczos.stochastic_quadrature(A, nv, 0, 0, 0, lanczos_deg, 1e-7, 0, n, 1)
  assert quad_nw.shape[0] == nv * lanczos_deg 
  assert np.all(~np.isnan(quad_nw))

  ## Ensure we can recover the trace under deterministic settings
  quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=0, orth=0, num_threads=1)
  quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
  quad_est = (n / nv) * np.sum(quad_ests)
  assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.02) ## ensure trace estimate within 1% 
  
  ## Ensure multi-threading works
  quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=6, orth=0, num_threads=8)
  quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
  quad_est = (n / nv) * np.sum(quad_ests)
  assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.02), "Multithreaded quadrature is worse" ## ensure trace estimate within 2% for multi-threaded (different rng!)

  ## Ensure it's generally pretty close
  for _ in range(50):
    quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=-1, orth=0, num_threads=4)
    quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
    quad_est = (n / nv) * np.sum(quad_ests)
    assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.05)

  ## Try to achieve a decently high accuracy
  nv, lanczos_deg = 2500, 20
  quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=-1, orth=5, num_threads=8)
  quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
  quad_est = (n / nv) * np.sum(quad_ests)
  assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.01) ## Ensure we can get within 0.25% of true trace 

  # from bokeh.plotting import show
  # from primate.plotting import figure_trace
  # show(figure_trace(n * quad_ests))

## NOTE: trace estimation only works with isotropic vectors 
def test_slq_fixed():
  from sanity import girard_hutch
  np.random.seed(1234)
  n = 30
  ew = 0.2 + 1.5*np.linspace(0, 5, n)
  Q,R = np.linalg.qr(np.random.uniform(size=(n,n)))
  A = Q @ np.diag(ew) @ Q.T
  A = (A + A.T) / 2
  ew_true = np.linalg.eigvalsh(A)
  tr_est = girard_hutch(A, lambda x: x, nv = n, estimates=False)
  threshold = 0.05*(np.max(ew)*n - np.min(ew)*n)
  assert np.isclose(A.trace() - tr_est, 0.0, atol=threshold)

def test_slq_trace():
  from primate.trace import sl_trace, _lanczos
  np.random.seed(1234)
  A = gen_sym(30)
  n = A.shape[1]
  A = csc_array(A, dtype=np.float32)
  tr_est = sl_trace(A, maxiter = 20, num_threads=1)
  assert len(tr_est) == 20

def test_slq_trace_clt_atol():
  from primate.trace import sl_trace, _lanczos
  np.random.seed(1234)
  A = gen_sym(30)
  n = A.shape[1]
  A = csc_array(A, dtype=np.float32)
  
  from primate.stats import sample_mean_cinterval
  tr_est = sl_trace(A, nv = 100, num_threads=1, seed=5)
  ci = np.array([sample_mean_cinterval(tr_est[:i], sdist='normal') if i > 1 else [-np.inf, np.inf] for i in range(len(tr_est))])
  
  ## Detect when, for the fixed set of samples, the trace estimatro should converge by CLT 
  atol_threshold = (A.trace() * 0.05) / n
  clt_converged = np.ravel(0.5*np.diff(ci, axis=1)) <= atol_threshold
  assert np.any(clt_converged), "Did not converge!"
  converged_ind = np.flatnonzero(clt_converged)[0]

  ## Re-run with same seed and ensure the index matches
  tr_est = sl_trace(A, nv = 100, atol=atol_threshold, num_threads=1, seed=5)
  converged_online = np.take(np.flatnonzero(tr_est == 0.0), 0)
  assert converged_online == converged_ind, "SLQ not converging at correct index!"
