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
  A_sparse = csc_array(symmetric(n), dtype=np.float32)
  v0 = np.random.uniform(size=n).astype(np.float32)

  deg, rtol, orth = 6, 0.0, 0
  y_true = A_sparse @ v0
  y_test = approx_matvec(A_sparse, v0, deg)
  assert np.allclose(y_true, y_test, atol=1e-5)

  print(y_true)
  y_test_cpp = _lanczos.function_approx(A_sparse, v0, deg, rtol, orth, **dict(function="identity"))

  y_test_cpp = _lanczos.function_approx(A_sparse, v0, deg, rtol, orth, **dict(function="identity"))
  print(y_test_cpp)

  ## it's working, just scaling isn't right!
  # np.dot(y_true / np.linalg.norm(y_true))
  print(np.linalg.norm(y_true))
  print(np.linalg.norm(y_test_cpp))
