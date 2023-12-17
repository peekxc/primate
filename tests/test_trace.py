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

def test_trace_basic():
  n = 10 
  A = symmetric(n)
  from primate.trace import hutch, _trace
  tr_est = hutch(A, maxiter=100, rng="pcg64")
  tr_true = A.trace()
  assert np.isclose(tr_est, tr_true, atol=tr_true*0.05)
