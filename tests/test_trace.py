import numpy as np 
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse import csr_array

def test_trace_estimator():
  import imate
  from primate.trace import slq
  T = imate.toeplitz(np.random.uniform(20), np.random.uniform(19), gram=True)
  tr_est = slq(T, orthogonalize=0, confidence_level=0.95, error_rtol=1e-2, min_num_samples=150, max_num_samples=200, num_threads=1)
  tr_true = np.sum(T.diagonal())
  assert np.isclose(np.take(tr_est,0), tr_true, atol=np.abs(tr_true)*0.05), "Estimate is off more than 5%"


def test_numerical_rank():
  from primate.trace import slq
  A = np.random.uniform(size=(100,100))
  A = A.T @ A
  u, s, vt = np.linalg.svd(A)
  B = u[:,:15] @ np.diag(s[:15]) @ vt[:15,:]
  tr_est = slq(csr_array(B), matrix_function = "rank", threshold=1e-2, orthogonalize=3, confidence_level=0.95, error_rtol=1e-2, min_num_samples=150, max_num_samples=200, num_threads=1)
  tr_true = np.linalg.matrix_rank(B)
  # assert info['convergence']['converged'], "trace didn't converge"
  assert np.isclose(np.float64(tr_est), tr_true, atol=np.abs(tr_true)*0.05), "Estimate is off more than 5%"


# def test_eigen():
#   # %% test lanczos tridiagonalization
#   import numpy as np 
#   import primate
#   from scipy.linalg import eigh_tridiagonal
#   from scipy.sparse.linalg import eigsh
#   from scipy.sparse import random
#   n = 30 
#   A = random(n,n,density=0.30**2)
#   A = A @ A.T
#   alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
#   v0 = np.random.uniform(size=A.shape[1])
#   primate._sparse_eigen.lanczos_tridiagonalize(A, v0, 1e-9, n-1, alpha, beta)
#   ew_lanczos = np.sort(eigh_tridiagonal(alpha, beta[:-1], eigvals_only=True))[1:]
#   ew_true = np.sort(eigsh(A, k=n-1, return_eigenvectors=False))
#   assert np.mean(np.abs(ew_lanczos - ew_true)) <= 1e-5

# def test_operators():
#   # %% Test diagonal operator 
#   from primate import _operators
#   y = np.random.uniform(size=10).astype(np.float32)
#   op = _operators.PyDiagonalOperator(y)
#   assert np.allclose(op.matvec(y), y * y)
#   assert op.dtype == np.dtype(np.float32)
#   assert op.shape == (10,10)
#   assert isinstance(aslinearoperator(op), LinearOperator)

#   # %% Test dense operator 
#   A = np.random.uniform(size=(10, 10)).astype(np.float32)
#   op = _operators.PyDenseMatrix(A)
#   assert np.allclose(op.matvec(y), A @ y)
#   assert op.dtype == np.dtype(np.float32)
#   assert op.shape == (10,10)
#   assert isinstance(aslinearoperator(op), LinearOperator)

#   # %% Test linear operator 
#   # TODO: Implement matmat and rmatmat and maybe adjoint! Should fix the svd issue
#   from primate import _operators
#   y = np.random.uniform(size=10).astype(np.float32)
#   A = np.random.uniform(size=(10, 10)).astype(np.float32)
#   A = A @ A.T
#   lo = aslinearoperator(A)
#   op = aslinearoperator(_operators.PyLinearOperator(lo))
#   for _ in range(100):
#     y = np.random.uniform(size=10).astype(np.float32)
#     assert np.allclose(op.matvec(y), A @ y)
#   assert op.dtype == np.dtype(np.float32)
#   assert op.shape == (10,10)
#   assert isinstance(aslinearoperator(op), LinearOperator)
#   from scipy.sparse.linalg import eigsh
#   np.allclose(np.abs(eigsh(op, k = 9)[0]), np.abs(eigsh(A, k = 9)[0]), atol=1e-6)

#   # %% Test adjoint operator 
#   # TODO: Implement matmat and rmatmat and maybe adjoint! Should fix the svd issue
#   from primate import _operators
#   y = np.random.uniform(size=10).astype(np.float32)
#   A = np.random.uniform(size=(10, 10)).astype(np.float32)
#   lo = aslinearoperator(A)
#   op = _operators.PyAdjointOperator(lo)
#   for _ in range(100):
#     y = np.random.uniform(size=10).astype(np.float32)
#     assert np.allclose(op.matvec(y), A @ y)
#     assert np.allclose(op.rmatvec(y), A.T @ y)
#   y = np.array([-0.61796093, 0.0286501, -0.1391555, -0.11632636, 0.15005986, -0.17718734, 0.21373063, -0.38565466, 0.3330924, 0.16717383])
#   op.matvec(y), A @ y
#   assert op.dtype == np.dtype(np.float32)
#   assert op.shape == (10,10)
#   assert isinstance(aslinearoperator(op), LinearOperator) ## todo: why isn't this true generically? gues SciPy doesn't use abc's 
#   from scipy.sparse.linalg import svds
#   np.abs(svds(op, k=9, tol=0)[1]) 
#   np.abs(svds(lo, k=9, tol=0)[1])
#   np.abs(svds(A, k=9, tol=0)[1])
#   ## How are these not identical???

#   ## See: https://github.com/scipy/scipy/issues/16928
#   class MyLinearOp:
#     def __init__(self, op):
#       self.op = op
#       self.dtype = A.dtype
#       self.shape = A.shape
    
#     def matvec(self, v: np.ndarray) -> np.ndarray: 
#       v = np.ravel(v) 
#       assert len(v) == self.shape[1]
#       x = self.op.matvec(v)
#       if not np.allclose(x, A @ v):
#         print(np.linalg.norm(x-v))
#       #assert np.allclose(x, A @ v)
#       return x
    
#     def rmatvec(self, v: np.ndarray) -> np.ndarray: 
#       v = np.ravel(v) 
#       assert len(v) == self.shape[0]
#       x = self.op.rmatvec(v)
#       if not np.allclose(x, A @ v):
#         # print(np.linalg.norm(x-v))
#         print(v)
#       #assert np.allclose(x, A.T @ v)
#       return x
    
#     def _matmat(self, X):
#       return np.column_stack([self.matvec(col) for col in X.T])

#     def _rmatmat(self, X):
#       return np.column_stack([self.rmatvec(col) for col in X.T])  

#     my_op = MyLinearOp(op)
#     np.abs(svds(my_op, k=9, tol=0)[1])
#     np.abs(svds(MyLinearOp(aslinearoperator(A)), k=9, tol=0)[1])



#   # %% test bidiagonalization
#   from primate import _diagonalize
#   v0 = np.random.uniform(size=op.shape[1])
#   lanczos_tol = 0.01 
#   orthogonalize = 3
#   alpha, beta = np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)
#   _diagonalize.golub_kahan_bidiagonalize(op, v0, lanczos_tol, orthogonalize, alpha, beta)

#   # %% Test Lanczos via Diagonal operator C++ 
#   from primate import _diagonalize
#   d = np.random.uniform(size=10, low=0, high=1).astype(np.float32)
#   v0 = np.random.uniform(size=10, low=-1, high=1).astype(np.float32)
#   alpha, beta = np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)
#   _diagonalize.test_lanczos(d, v0, 1e-8, 0, alpha, beta)
#   V = np.zeros(shape=(10,10), dtype=np.float32)
#   _diagonalize.eigh_tridiagonal(alpha, beta, V) # note this replaces alphas
#   np.allclose(np.sort(d), np.sort(alpha), atol=1e-6) 

# def test_
# %%
