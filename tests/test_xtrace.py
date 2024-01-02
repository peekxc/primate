import numpy as np
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.trace import xtrace
import primate.random as random 

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

def test_xtrace_trace():
  np.random.seed(1234)
  n = 100
  A = csc_array(random.symmetric(n, pd = True), dtype=np.float32)
  assert np.isclose(A.trace(), xtrace(A), atol=1e-5)

def test_xtrace_mf():
  np.random.seed(1234)
  n = 100
  A = random.symmetric(n, pd = True)
  ew, ev = np.linalg.eigh(A)
  for fun_name, fun in zip(["identity", "log", "inv", "exp"], [lambda x: x, np.log, np.reciprocal, np.exp]):
    trace_test = xtrace(A, fun=fun)
    trace_true = (ev @ np.diag(fun(ew)) @ ev.T).trace()
    assert np.isclose(trace_test, trace_true, atol=abs(0.001*trace_true))

