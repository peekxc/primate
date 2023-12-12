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

def test_vector_approx():
  ## theta size desn't match 
  from primate.diagonalize import _lanczos
  from sanity import approx_matvec
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n, psd = True), dtype=np.float32)
  v0 = np.random.uniform(size=n).astype(np.float32)
  deg, rtol, orth = 6, 0.0, 0

  ## Test that very basic implementation works as expected on identity
  y_true = A_sparse @ v0
  y_test = approx_matvec(A_sparse, v0, deg)
  assert np.allclose(y_true, y_test, atol=1e-5)

  ## Test more efficient implementation works
  y_test_cpp = _lanczos.function_approx(A_sparse, v0, deg, rtol, orth, **dict(function="identity"))
  assert np.max(np.abs(y_test_cpp - y_true)) < 1e-5 

def test_mf_approx():
  from primate.diagonalize import _lanczos
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n, psd = True), dtype=np.float32)
  deg, rtol, orth = 6, 0.0, 0
  M = _lanczos.MatrixFunction_sparse(A_sparse, deg, rtol, orth, **dict(function="identity"))
  
  ## Basic checks 
  assert M.shape == (10, 10)
  v0 = np.random.normal(size=M.shape[0])
  y_test = M.matvec(v0)
  assert isinstance(y_test, np.ndarray)
  assert np.max(np.abs(y_test - A_sparse @ v0)) < 1e-5

  ## Check matrix functions
  ew, ev = np.linalg.eigh(A_sparse.todense()) 
  M = _lanczos.MatrixFunction_sparse(A_sparse, deg, rtol, orth, **dict(function="log"))
  y_log_test = M.matvec(v0)
  y_log_true = (ev @ np.diag(np.log(ew)) @ ev.T) @ v0
  assert np.max(np.abs(M.matvec(v0) - y_log_true)) <= 0.05
  assert np.all(y_log_test != y_test)

def test_mf_api():
  assert True

