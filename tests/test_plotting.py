import numpy as np 
from primate.plotting import figure_trace

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')


def test_plotting():
  


