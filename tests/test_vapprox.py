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

def test_mf_basic():
  from primate.diagonalize import _lanczos
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n, psd = True), dtype=np.float32)
  deg, rtol, orth = 6, 0.0, 0
  M = _lanczos.MatrixFunction_sparseF(A_sparse, deg, rtol, orth, **dict(function="identity"))
  
  ## Basic checks 
  assert M.shape == (10, 10)
  v0 = np.random.normal(size=M.shape[0])
  y_test = M.matvec(v0)
  assert isinstance(y_test, np.ndarray)
  assert np.max(np.abs(y_test - A_sparse @ v0)) < 1e-5


def test_mf_api():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n, psd = True), dtype=np.float32)
  M = matrix_function(A_sparse)
  v0 = np.random.normal(size=n)
  assert np.max(np.abs(M.matvec(v0) - A_sparse @ v0)) <= 1e-6, "MF matvec doesn't match identity"
  assert M.dtype == np.float32, "dtype mismatch"

  from scipy.sparse.linalg import eigsh
  ew_true = eigsh(A_sparse)[0]
  ew_test = eigsh(M)[0]
  assert np.allclose(ew_true, ew_test, atol=1e-5), "eigenvalues mismatch / operator doesn't register as respecting LO interface"

def test_mf_matmat():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n, psd = True), dtype=np.float32)
  v = np.random.uniform(size=n).astype(np.float32)
  M = matrix_function(A, "identity")
  assert hasattr(M, "matmat") and hasattr(M, "__matmul__")
  v = np.random.normal(size=M.shape[1])
  v_out1 = M.matvec(v)
  v_out2 = M.matmat(v)
  v_out3 = M @ v[:,np.newaxis]
  assert np.allclose(v_out1, A.dot(v), atol=1e-8)
  # np.allclose(v_out2, A @ v, atol=1e-8)
  assert np.allclose(v_out3, A @ v[:,np.newaxis], atol=1e-8)

  V = np.random.uniform(size=(n, 5)).astype(np.float32)
  assert (M @ V).shape == (10, 5)


def test_mf_approx():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n, psd = True), dtype=np.float32)
  v = np.random.uniform(size=n).astype(np.float32)

  ## Get the eigenvalues and eigenvectors
  ew, ev = np.linalg.eigh(A.todense()) 
  for fun_name, fun in zip(["identity", "log", "inv", "exp"], [lambda x: x, np.log, np.reciprocal, np.exp]):
    M = matrix_function(A, fun_name)
    y_test = M.matvec(v)
    y_true = (ev @ np.diag(fun(ew)) @ ev.T) @ v
    assert np.max(np.abs(y_test - y_true)) <= 1e-4


# class CustomOperator:
#   def __init__(self, A):
#     self.A = A
#     self.dtype = A.dtype
#     self.shape = A.shape
#   # def _matvec(self, v):
#   #   return self.A @ v
#   def matvec(self, v):
#     return self.A @ v
#   # def _matmat(self, V):
#   #   return self.A @ V
#   def matmat(self, V):
#     return self.A @ V

# A_op = CustomOperator(A_sparse)
