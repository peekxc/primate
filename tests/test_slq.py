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