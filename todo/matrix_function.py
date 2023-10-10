

# %% Imate API 
import imate 
from imate.linear_operator import LinearOperator, Matrix, AffineMatrixFunction, linear_operator

#LinearOperator
# self.Aop = None
# self.cpu_Aop = None
# self.gpu_Aop = None
# self.gpu = False
# self.num_gpu_devices = 0
# self.initialized_on_cpu = False
# self.initialized_on_gpu = False
# self.data_type_name = None
# self.num_parameters = None


trace_est()


# %% 
from scipy.sparse import random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
A = random(50, 50, density=0.05)
A = aslinearoperator(A)
imate.trace(A)
imate.linear_operator.LinearOperator


# %% How to infer the rank from ||A + t*I||^p for varying t and fixed p? 
import numpy as np 
ew_nonz = np.random.uniform(size=40, low=0.05, high=2.0)
ew_zero = np.random.uniform(size=15, low=0.0, high=1e-9)
ew = np.append(ew_zero, ew_nonz)

import bokeh 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import Range1d
output_notebook()

timepoints = np.geomspace(1e-16, 1.0, 1500)
ew_traces = np.array([np.sum((ew + t)**0.50) for t in timepoints])
p = figure(width=200, height=200, x_axis_type='log')
p.line(timepoints, ew_traces)
p.x_range = Range1d(1e-10, 10.0)
show(p) 


# %% Test ortho 
from pyimate.trace import trace
# from pyimate.ortho import _ortho
from pyimate._random_generator import * 


# %% Test random 
import numpy as np
from pyimate import rademacher

## Indeed it is fast! 
import timeit
timeit.timeit(lambda: np.random.choice([-1.0, +1.0], size=5000), number=10000)
timeit.timeit(lambda: rademacher(5000), number=10000)

_random_generator
_random_generator.rademacher(10)


# %% Test diagonalization -- it works!
# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh_tridiagonal.html 
import numpy as np 
d = 3*np.ones(4)
e = -1*np.ones(3)
A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1) # 
from scipy.linalg import eigh_tridiagonal

from pyimate import _diagonalize
V = np.zeros((len(d), len(d)), dtype=np.float32)
_diagonalize.eigh_tridiagonal(d, e, V)
V

# %% Test orthogonalize 
from pyimate import _orthogonalize
import numpy as np 
X = np.random.uniform(size=(5,5)).astype(dtype=np.float32)

np.linalg.norm(X, axis=1)
_orthogonalize.orthogonalize_vectors(X)
assert np.all(np.abs((X @ X.T) - np.diag(np.diag(X @ X.T))) <= 1e-6)


# %% Test operators 
import numpy as np 
from pyimate import _operators
y = np.random.uniform(size=10).astype(np.float32)
op = _operators.PyDiagonalOperator(y)
np.allclose(op.matvec(y), y * y)


# TODO: once you can define a LinearOperator from python, then try https://github.com/scipy/scipy/blob/v1.11.2/scipy/integrate/__quadpack.h
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse import random
L = aslinearoperator(random(10, 10, density=0.4))
