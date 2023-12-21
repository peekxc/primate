import numpy as np 
from numbers import Number
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric
from typing import * 

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

## NOTE: trace estimation only works with isotropic vectors 
def test_girard_fixed():
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
  assert np.allclose(ew_true, ew)
  assert np.isclose(A.trace() - tr_est, 0.0, atol=threshold)

def test_trace_import():
  import primate.trace
  assert '_trace' in dir(primate.trace)
  from primate.trace import hutch, _trace
  assert 'hutch' in dir(_trace)
  assert isinstance(hutch, Callable)

def test_trace_basic():
  from primate.trace import hutch
  np.random.seed(1234)
  n = 10 
  A = symmetric(n)
  tr_test1 = hutch(A, maxiter=100, seed=5, num_threads=1)
  tr_test2 = hutch(A, maxiter=100, seed=5, num_threads=1)
  tr_true = A.trace()
  assert tr_test1 == tr_test2, "Builds not reproducible!"
  assert np.isclose(tr_test1, tr_true, atol=tr_true*0.05)

def test_trace_inputs():
  from primate.trace import hutch
  n = 10 
  A = symmetric(n)
  tr_1 = hutch(A, maxiter=100)
  tr_2 = hutch(csc_array(A), maxiter=100)
  tr_3 = hutch(aslinearoperator(A), maxiter=100)
  assert all([isinstance(t, Number) for t in [tr_1, tr_2, tr_3]]) 

def test_hutch_info():
  from primate.trace import hutch
  np.random.seed(1234)
  n = 25
  A = csc_array(symmetric(n), dtype=np.float32)
  tr_est, info = hutch(A, maxiter = 200, info=True)
  assert isinstance(info, dict) and isinstance(tr_est, Number)
  assert len(info['samples']) == 200
  assert np.all(~np.isclose(info['samples'], 0.0))
  assert np.isclose(tr_est, A.trace(), atol=1.0)

def test_hutch_multithread():
  from primate.trace import hutch
  np.random.seed(1234)
  n = 25
  A = csc_array(symmetric(n), dtype=np.float32)
  tr_est, info = hutch(A, maxiter = 200, atol=0.0, info = True, num_threads=6)
  assert len(info['samples'] == 200)
  assert np.all(~np.isclose(info['samples'], 0.0))
  assert np.isclose(tr_est, A.trace(), atol=1.0)

def test_hutch_clt_atol():
  from primate.trace import hutch
  np.random.seed(1234)
  n = 30
  A = csc_array(symmetric(n), dtype=np.float32)
  
  from primate.stats import sample_mean_cinterval
  tr_est, info = hutch(A, maxiter = 100, num_threads=1, seed=5, info=True)
  tr_samples = info['samples']
  ci = np.array([sample_mean_cinterval(tr_samples[:i], sdist='normal') if i > 1 else [-np.inf, np.inf] for i in range(len(tr_samples))])
  
  ## Detect when, for the fixed set of samples, the trace estimator should converge by CLT 
  atol_threshold = (A.trace() * 0.05)
  clt_converged = np.ravel(0.5*np.diff(ci, axis=1)) <= atol_threshold
  assert np.any(clt_converged), "Did not converge!"
  converged_ind = np.flatnonzero(clt_converged)[0]

  ## Re-run with same seed and ensure the index matches
  tr_est, info = hutch(A, maxiter = 100, num_threads=1, atol=atol_threshold, seed=5, info=True)
  tr_samples = info['samples']
  converged_online = np.take(np.flatnonzero(tr_samples == 0.0), 0)
  assert converged_online == converged_ind, "hutch not converging at correct index!"

def test_hutch_change():
  from primate.trace import hutch
  np.random.seed(1234)
  n = 30
  A = csc_array(symmetric(n), dtype=np.float32)
  tr_est, info = hutch(A, maxiter = 100, num_threads=1, seed=5, info=True)
  tr_samples = info['samples']
  estimator = np.cumsum(tr_samples) / np.arange(1, 101)
  conv_ind_true = np.flatnonzero(np.abs(np.diff(estimator)) <= 0.001)[0] + 1

  ## Test the convergence checking for the atol change method
  tr_est, info = hutch(A, maxiter = 100, num_threads=1, seed=5, info=True, atol=0.001, stop="change")
  conv_ind_test = np.take(np.flatnonzero(info['samples'] == 0), 0)
  assert abs(conv_ind_true - conv_ind_test) <= 1

def test_trace_mf():
  from primate.trace import hutch
  n = 10 
  A = symmetric(n)
  tr_est = hutch(A, fun="identity", maxiter=100, num_threads=1, seed = 5)
  tr_true = A.trace()
  assert np.isclose(tr_est, tr_true, atol=tr_true*0.05)
  
  ## TODO 
  # tr_est, info = hutch(A, fun=lambda x: x, maxiter=100, seed=5, num_threads=1, info=True)
  # assert np.isclose(tr_est, tr_true, atol=tr_true*0.05)
  # for s in range(15000):
  #   est, info = hutch(A, fun="identity", deg=2, maxiter=200, num_threads=1, seed=591, info=True)
  #   assert not np.isnan(est)

  # from primate.operator import matrix_function
  # M = matrix_function(A, fun="identity", deg=20)
  # for s in range(15000):
  #   v0 = np.random.choice([-1, 1], size=M.shape[0])
  #   assert not np.isnan(M.quad(v0))
  #   M.krylov_basis
  #   M.ncv
  #   M.deg
  #   M._alpha
  #   M._beta
  #   M.matvec(-v0)

  # from primate.diagonalize import lanczos
  # lanczos(A, v0=v0, rtol=M.rtol, deg=M.deg, orth=M.orth)
    # if np.any(np.isnan(info['samples']))