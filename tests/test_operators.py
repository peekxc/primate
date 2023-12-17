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

def test_dense_operator():
  from primate.operator import _operators
  assert "DenseMatrix" in dir(_operators)
  A = np.random.uniform(size=(10,10))
  A_op = _operators.DenseMatrix(A)
  x, X = np.random.normal(size=10), np.random.normal(size=(10, 5))
  assert A_op.shape == (10,10)
  assert A_op.dtype == np.float32
  assert np.allclose(A_op.matvec(x), A.dot(x))
  assert np.allclose(A_op.matmat(X), A.dot(X))
  assert np.allclose(A_op @ X, A @ X)
  # assert np.allclose(A_op @ x, A @ x)

def test_sparse_operator():
  from primate.operator import _operators
  assert "SparseMatrix" in dir(_operators)
  A = csc_array(np.random.uniform(size=(10,10)))
  A_op = _operators.SparseMatrix(A)
  x, X = np.random.normal(size=10), np.random.normal(size=(10, 5))
  assert A_op.shape == (10,10)
  assert A_op.dtype == np.float32
  assert np.allclose(A_op.matvec(x), A.dot(x))
  assert np.allclose(A_op.matmat(X), A.dot(X))
  assert np.allclose(A_op @ X, A @ X)

