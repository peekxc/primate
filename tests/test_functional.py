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

def test_functional():
  from primate.functional import numrank
  np.random.seed(1234)
  u = np.random.uniform(size=100, low=0, high=1)
  u[u < 0.20] = 0.0
  A = symmetric(100, pd=False, ew=u)
  true_rank = np.linalg.matrix_rank(A)
  correct = np.zeros(30, dtype=bool)
  for i in range(30):
    # est = hutch(A, fun="numrank", maxiter=15000, threshold=min_ew_est, stop="change", atol=0.001, info=False, verbose=False)
    correct[i] = numrank(A) == true_rank
  assert np.sum(correct) >= 25
  # import timeit
  # timeit.timeit(lambda: numrank(A, est="xtrace"), number = 30)
  # timeit.timeit(lambda: np.linalg.matrix_rank(A), number=30)
  # print(f"Intended rank: {np.sum(u != 0.0)}")
  # print(f"Sum >= t:      {sum(ew >= min_ew_est*0.95)}")
  # print(f"Actual rank:   {np.linalg.matrix_rank(A)}")

def test_rank_time():
  from primate.operator import Toeplitz
  from primate.functional import numrank
  ind = np.arange(100)
  T = Toeplitz(ind)
  assert numrank(T) == 100
