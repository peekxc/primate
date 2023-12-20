from primate.plotting import figure_trace
from primate.random import symmetric
from primate.trace import hutch

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

def test_plotting():
  from bokeh.models.layouts import Row
  n = 10 
  A = symmetric(n)
  tr_est, info = hutch(A, fun="identity", maxiter=100, num_threads=1, seed=5, info=True)
  p = figure_trace(info["samples"])
  assert isinstance(p, Row)