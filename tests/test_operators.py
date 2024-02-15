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
  A_sparse = csc_array(symmetric(n, pd = True), dtype=np.float32)
  v0 = np.random.uniform(size=n).astype(np.float32)
  deg, rtol, orth = 6, 0.0, 0

  ## Test that very basic implementation works as expected on identity
  y_true = A_sparse @ v0
  y_test = approx_matvec(A_sparse, v0, deg)
  assert np.allclose(y_true, y_test, atol=1e-5)

def test_mf_fun():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10 
  A = np.eye(n)
  M = matrix_function(A, fun=lambda x: x)
  x, y = np.arange(10)+1, np.arange(10)+1
  assert np.allclose(M.fun(x), y)
  M = matrix_function(A, fun=np.exp)
  assert np.allclose(M.fun(x), np.exp(y))
  M = matrix_function(A, fun="exp")
  assert np.allclose(M.fun(x), np.exp(y))
  # from math import exp
  # M = matrix_function(A, fun=exp)
  # M.fun([0,1,2,3])
  # M.fun(list(np.arange(M.shape[0])))
  # M.fun(range(10))
  # M.fun(np.array([0]*10))
  # M.fun(np.arange(M.shape[0]))



def test_mf_basic():
  from primate.operator import _operators
  np.random.seed(1234)
  n = 10
  A = symmetric(n, pd = True).astype(np.float32)
  deg, rtol, orth = 6, 0.0, 0
  M = _operators.DenseF_MatrixFunction(A, deg, rtol, orth, 0, **dict(function="identity"))
  
  ## Basic checks 
  assert M.shape == (10, 10)
  v0 = np.random.normal(size=M.shape[0])
  y_true = A @ v0
  y_test = M.matvec(v0)
  assert isinstance(y_test, np.ndarray)
  assert np.max(np.abs(y_test - y_true)) < 1e-5

def test_mf_quad():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 15
  A = symmetric(n, pd = True).astype(np.float32)
  M = matrix_function(A, fun='identity')

  ## M quad matches v0 when v0 has norm 1 
  v0 = np.random.choice([-1.0, +1.0], size=M.shape[0]) / np.sqrt(n)
  assert not np.isnan(M.quad(v0))
  assert np.isclose(v0 @ A @ v0, M.quad(v0))
  assert isinstance(M.krylov_basis, np.ndarray) and np.all(~np.isnan(M.krylov_basis))
  assert np.all(~np.isnan(M.nodes)) and np.all(~np.isnan(M.weights))

  ## Test non-unit rademacher vector inputs
  v0 = np.random.choice([-1.0, +1.0], size=n)
  assert not np.isnan(M.quad(v0))
  assert np.isclose(v0 @ A @ v0, M.quad(v0))

  ## Test totally random vectors
  v0 = np.random.uniform(low=-1.0, high=1.0, size=n)
  assert not np.isnan(M.quad(v0))
  assert np.isclose(v0 @ A @ v0, M.quad(v0))

  ## Ensure the first column vector of the Krylov basis is v  
  assert np.allclose((v0 / np.linalg.norm(v0)), M.krylov_basis[:,0])
  
def test_mf_trace_quad():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A = symmetric(n, pd = True).astype(np.float32)
  M = matrix_function(A, fun='identity')
  tr_true = A.trace()

  tr_est = [M.quad(np.random.choice([-1.0, +1.0], size=M.shape[0])) for i in range(500)]
  assert np.isclose(tr_true, np.mean(tr_est), atol=tr_true*0.05)

  tr_est = [M.quad(np.random.choice([-1.0, +1.0], size=M.shape[0])/np.sqrt(10)) for i in range(500)]
  assert np.isclose(tr_true, np.mean(tr_est)*n, atol=tr_true*0.05)

  ## As long as unit norm 0-mean vectors are passed in, multiply by n yields trace estimate
  tr_est = [M.quad((v := np.random.normal(size=M.shape[0]))/np.linalg.norm(v)) for i in range(500)]
  assert np.isclose(tr_true, np.mean(tr_est)*n, atol=tr_true*0.05)

  ## Scaling is now handled automatically
  tr_est = [M.quad((v := np.random.normal(size=M.shape[0]))) for i in range(500)]
  assert np.isclose(tr_true, np.mean(tr_est), atol=tr_true*0.05)

def test_mf_api():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n, pd = True), dtype=np.float32)
  M = matrix_function(A_sparse)
  v0 = np.random.normal(size=n)
  assert np.max(np.abs(M.matvec(v0) - A_sparse @ v0)) <= 1e-5, "MF matvec doesn't match identity"
  assert M.dtype == np.float32, "dtype mismatch"
  assert np.allclose(M.fun(np.zeros(10)), np.zeros(10))
  assert M.shape == (10, 10)

def test_mf_pyfun():
  from primate.operator import matrix_function
  from primate.operator import _operators
  np.random.seed(1234)
  n = 10
  A = symmetric(n, pd = True).astype(np.float32)
  M = matrix_function(A, fun="log") 
  assert np.allclose(M.fun(np.repeat(1.0, 10)), np.zeros(10))
  assert M.native_fun
  M = matrix_function(A, fun=lambda x: np.log(x))
  assert M.fun(1.0) == 0.0 
  assert not(M.native_fun)
  M.fun = "exp"
  assert np.isclose(M.fun(np.ones(1)), np.exp(1.0))
  assert M.native_fun
  M.fun = lambda x: x
  assert M.fun(np.ones(1)) == 1.0
  assert not(M.native_fun)
  M.fun = np.exp
  assert np.isclose(M.fun(np.ones(1)), np.exp(1.0))
  assert not(M.native_fun)
  assert not(matrix_function(A, fun=np.log).native_fun)

  ## **New** : infer stateless native C ptrs! 
  M.fun = _operators.exp
  assert M.native_fun, "failed to infer native function!"

  # assert np.isclose(M.fun(np.ones(1)), np.exp(1.0))
  
  ## TODO: figure out kwargs, add ability for raw function ptrs in trace interface!
  # M.fun = (M, "exp", dict(t=1))
  # setattr(M.f)
  # M.fun.__init__("exp", t = 2.0)
  # M.fun(10) np.exp(-10)

def test_mf_eigen():
  from scipy.sparse.linalg import eigsh
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n, pd = True), dtype=np.float32)
  M = matrix_function(A_sparse)
  ew_true = eigsh(A_sparse)[0]
  ew_test = eigsh(M)[0]
  assert np.allclose(ew_true, ew_test, atol=1e-5), f"eigenvalues mismatch error = {np.max(np.abs(ew_true - ew_test)):0.5f}/ operator doesn't register as respecting LO interface"

def test_mf_deflate():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n, pd = True), dtype=np.float32)
  M = matrix_function(A, fun=np.exp)
  Q = np.linalg.qr(np.random.normal(size=(10, 5)))[0]
  r = np.random.normal(size=10)
  assert np.allclose(r, M.transform(r))
  x = M.matvec(r)
  M.deflate(Q)
  assert ~np.allclose(r, M.transform(r)), "No deflation happening"
  assert np.allclose(M.transform(r), r - Q @ (Q.T @ r)), "Deflation step failed"
  z = M.matvec(r)
  assert not(np.allclose(x, z))
  
  
def test_mf_matmat():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n, pd = True), dtype=np.float32)
  v = np.random.uniform(size=n).astype(np.float32)
  M = matrix_function(A, "identity")
  assert hasattr(M, "matmat") and hasattr(M, "__matmul__")
  v, V = np.random.normal(size=M.shape[1]), np.random.uniform(size=(n, 5)).astype(np.float32)
  v_out1 = M.matvec(v)
  v_out2 = M.matmat(v)
  v_out3 = M @ v[:,np.newaxis]
  assert np.allclose(v_out1, A.dot(v), atol=1e-8)
  # np.allclose(v_out2, A @ v, atol=1e-8)
  assert np.allclose(v_out3, A @ v[:,np.newaxis], atol=1e-8)
  assert (M @ V).shape == (10, 5)

def test_mf_approx():
  from primate.operator import matrix_function
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n, pd = True), dtype=np.float32)
  v = np.random.uniform(size=n).astype(np.float32)

  ## Get the eigenvalues and eigenvectors
  ew, ev = np.linalg.eigh(A.todense()) 
  for fun_name, fun in zip(["identity", "log", "inv", "exp"], [lambda x: x, np.log, np.reciprocal, np.exp]):
    M = matrix_function(A, fun_name)
    y_test = M.matvec(v)
    y_true = (ev @ np.diag(fun(ew)) @ ev.T) @ v
    assert np.max(np.abs(y_test - y_true)) <= 1e-4

def test_mf_trace():
  from primate.operator import matrix_function
  from primate.random import rademacher
  np.random.seed(1234)
  n = 10
  A = symmetric(n, pd = True).astype(np.float32)
  ew = np.linalg.eigvalsh(A)
  for fun, f in zip(["identity", "inv", "log", "exp"], [lambda x: x, np.reciprocal, np.log, np.exp]):
    M = matrix_function(A, fun=fun)
    tr_true = np.sum(f(ew))
    tr_test = np.mean([M.quad(rademacher(size=n, seed=s+1)) for s in range(500)])
    assert np.isclose(tr_true, tr_test, atol=abs(tr_true * 0.05))

def test_mf_quad_method():
  from primate.operator import matrix_function
  from primate.random import rademacher
  np.random.seed(1234)
  n = 100
  A = symmetric(n, pd = True).astype(np.float32)
  M = matrix_function(A, fun="identity")
  for _ in range(30):
    M.method = "golub_welsch"
    assert M.method == "golub_welsch"
    # v = rademacher(size=100)
    v = np.random.normal(size=100, loc=0, scale=10)
    gh_quad = M.quad(v)
    M.method = "fttr"
    assert M.method == "fttr"
    ft_quad = M.quad(v)
    assert np.isclose(ft_quad, gh_quad, atol=abs(0.01*gh_quad))

# def test_mf_transform():
#   from primate.operator import matrix_function
#   from primate.random import rademacher
#   np.random.seed(1234)
#   n = 100
#   A = symmetric(n, pd = True).astype(np.float32)
#   M = matrix_function(A, fun="identity")
  
#   x = np.array([0.1, 0.2])
#   M.transform(x.data)



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

# def test_dense_operator():
#   from primate.operator import _operators
#   assert "DenseMatrix" in dir(_operators)
#   A = np.random.uniform(size=(10,10))
#   A_op = _operators.DenseMatrix(A)
#   x, X = np.random.normal(size=10), np.random.normal(size=(10, 5))
#   assert A_op.shape == (10,10)
#   assert A_op.dtype == np.float32
#   assert np.allclose(A_op.matvec(x), A.dot(x))
#   assert np.allclose(A_op.matmat(X), A.dot(X))
#   assert np.allclose(A_op @ X, A @ X)
#   # assert np.allclose(A_op @ x, A @ x)

# def test_sparse_operator():
#   from primate.operator import _operators
#   assert "SparseMatrix" in dir(_operators)
#   A = csc_array(np.random.uniform(size=(10,10)))
#   A_op = _operators.SparseMatrix(A)
#   x, X = np.random.normal(size=10), np.random.normal(size=(10, 5))
#   assert A_op.shape == (10,10)
#   assert A_op.dtype == np.float32
#   assert np.allclose(A_op.matvec(x), A.dot(x))
#   assert np.allclose(A_op.matmat(X), A.dot(X))
#   assert np.allclose(A_op @ X, A @ X)