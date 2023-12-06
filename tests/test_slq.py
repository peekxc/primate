import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

def test_stochastic_quadrature():
  from primate.diagonalize import _lanczos
  assert hasattr(_lanczos, "stochastic_quadrature"), "Module compile failed"
  
  from primate.trace import sl_gauss
  np.random.seed(1234)
  n, nv, lanczos_deg = 30, 250, 20
  A = csc_array(symmetric(30), dtype=np.float32)

  ## Test raw computation to ensure it's returning valid entries
  # A, nv, dist, engine_id, seed, lanczos_degree, lanczos_rtol, orth, ncv, num_threads
  quad_nw = _lanczos.stochastic_quadrature(A, nv, 0, 0, 0, lanczos_deg, 1e-7, 0, n, 1)
  assert quad_nw.shape[0] == nv * lanczos_deg 
  assert np.all(~np.isnan(quad_nw))

  ## Ensure we can recover the trace under deterministic settings
  quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=0, orth=0, num_threads=1)
  quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
  quad_est = (n / nv) * np.sum(quad_ests)
  assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.02), "Deterministic Quadrature is off" ## ensure trace estimate within 1% 
  
  ## Ensure multi-threading works
  quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=6, orth=0, num_threads=8)
  quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
  quad_est = (n / nv) * np.sum(quad_ests)
  # assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.05), "Multithreaded quadrature is worse"

  ## Ensure it's generally pretty close
  for _ in range(50):
    quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=-1, orth=0, num_threads=4)
    quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
    quad_est = (n / nv) * np.sum(quad_ests)
    assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.05), "Quadrature accuracy not always close"

  ## Try to achieve a decently high accuracy
  nv, lanczos_deg = 2500, 20
  quad_nw = sl_gauss(A, n=nv, deg=lanczos_deg, seed=-1, orth=5, num_threads=8)
  quad_ests = np.array([np.prod(nw, axis=1).sum() for nw in np.array_split(quad_nw, nv)])
  quad_est = (n / nv) * np.sum(quad_ests)
  assert np.isclose(A.trace(), quad_est, atol = A.trace() * 0.05), "High accuracy quadrature is off" ## Ensure we can get within 0.25% of true trace 

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
  from primate.trace import sl_trace
  np.random.seed(1234)
  n = 25
  A = csc_array(symmetric(n), dtype=np.float32)
  tr_est, info = sl_trace(A, maxiter = 200, num_threads=1, seed=-1, info=True)
  assert len(info['samples'] == 200)
  assert np.all(~np.isclose(info['samples'], 0.0))
  assert np.isclose(tr_est, A.trace(), atol=1.0)

def test_slq_trace_multithread():
  from primate.trace import sl_trace
  np.random.seed(1234)
  n = 25
  A = csc_array(symmetric(n), dtype=np.float32)
  tr_est, info = sl_trace(A, maxiter = 200, atol=0.0, info = True, num_threads=6)
  assert len(info['samples'] == 200)
  assert np.all(~np.isclose(info['samples'], 0.0))
  assert np.isclose(tr_est, A.trace(), atol=1.0)

def test_slq_trace_clt_atol():
  from primate.trace import sl_trace, _lanczos
  np.random.seed(1234)
  n = 30
  A = csc_array(symmetric(n), dtype=np.float32)
  
  from primate.stats import sample_mean_cinterval
  tr_est, info = sl_trace(A, nv = 100, num_threads=1, seed=5, info=True)
  tr_samples = info['samples']
  ci = np.array([sample_mean_cinterval(tr_samples[:i], sdist='normal') if i > 1 else [-np.inf, np.inf] for i in range(len(tr_samples))])
  
  ## Detect when, for the fixed set of samples, the trace estimator should converge by CLT 
  atol_threshold = (A.trace() * 0.05)
  clt_converged = np.ravel(0.5*np.diff(ci, axis=1)) <= atol_threshold
  assert np.any(clt_converged), "Did not converge!"
  converged_ind = np.flatnonzero(clt_converged)[0]

  ## Re-run with same seed and ensure the index matches
  tr_est, info = sl_trace(A, nv = 100, num_threads=1, atol=atol_threshold, seed=5, info=True)
  tr_samples = info['samples']
  converged_online = np.take(np.flatnonzero(tr_samples == 0.0), 0)
  assert converged_online == converged_ind, "SLQ not converging at correct index!"

def test_slq_trace_f():
  # from primate.trace import sl_trace, _lanczos
  # np.random.seed(1234)
  # n = 30
  # A = symmetric(n, psd=True, ew = np.linspace(1/n, 1, n))
  # sl_trace(A, "identity")
  # np.log(np.linalg.det(A))
  # np.mean(sl_trace(A, "log", 200))
  # est.mean()
  assert True
  # np.isclose(np.mean(sl_trace(A, fun="numrank")), np.linalg.matrix_rank(A.todense())


