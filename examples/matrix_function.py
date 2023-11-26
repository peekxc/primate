# %% 
import numpy as np 
from primate.diagonalize import lanczos
from scipy.linalg import eigh_tridiagonal
from primate.random import rademacher
from typing import * 
from bokeh.plotting import show, figure
from bokeh.io import output_notebook
output_notebook()

# %% Start with a matrix with well-separated eigenvalues
np.random.seed(1234)
n = 30
# ew = 0.2 + 1.5*np.linspace(0, 5, n)
# ew = np.sort(np.random.uniform(low=0.2, high=7.0, size=n))
# ew = np.sort(np.random.normal(loc=0, scale=3, size=n))

## symmetric spectrum
ew = np.sort(np.random.normal(loc=0, scale=3, size=n // 2))
ew = np.sort(np.append(ew, -ew))
Q,R = np.linalg.qr(np.random.uniform(size=(n,n)))
A = Q @ np.diag(ew) @ Q.T
A = (A + A.T) / 2
assert np.allclose(np.linalg.eigvalsh(A) - ew, 0)

# %% Try to estimate the trace via Girard-Hutchinson estimator
sf = lambda x: x / (x + 1e-2)
trace_true = np.sum(sf(ew))

# Girard-Hutchinsen estimator 
# Note: indeed, just applying f to Rayleigh-Ritz values is not enough!
from primate.plotting import figure_trace
np.random.seed(1236)
# trace_estimates = np.array([girard_hutch(A, sf, orth=10, nv=1) for _ in range(200)])
trace_estimates = girard_hutch(A, sf, orth=10, nv=150, estimates=True)
show(figure_trace(trace_estimates, trace_true)) 

# %% Look at the set cumulative distributions of the quadrature nodes
# x_thresholds = np.linspace(0.95*min(ew), 1.05*max(ew), 100)
x_thresholds = np.linspace(0, 1, 100)
true_cdf = np.cumsum(ew / np.sum(ew))
true_cesm = np.array([np.sum(true_cdf < x) for x in x_thresholds])/A.shape[1]

## This sort of work but only depends on the Rayleigh-Ritz estimates actually
poly_deg = A.shape[1]
mean_cesm = np.zeros(len(x_thresholds))
p = figure(width=250, height=250)
for i in range(50):
  v0 = np.random.choice([-1.0, +1.0], size=A.shape[0]) 
  theta, tau = lanczos_quadrature(A, v0, k=poly_deg)
  # /poly_deg
  quadrature  = np.cumsum(theta*tau)
  test_cdf  = quadrature / np.max(quadrature)
  test_cesm = np.array([np.sum(test_cdf <= x) for x in x_thresholds]) / poly_deg
  mean_cesm += test_cesm
  p.line(x_thresholds, test_cesm, line_alpha = 0.25, line_color = 'black')
p.line(x_thresholds, true_cesm, color='blue', line_width=2)
p.line(x_thresholds, mean_cesm / 50, color='purple', line_width=1.5)
show(p)

# %% Indeed, one way to estimate the trace is via matrix-vector approximation
trace_true = A.trace()
trace_estimates = np.zeros(200)
for i in range(200):
  v0 = np.random.choice([-1, +1], size=A.shape[1])
  trace_estimates[i] = np.dot(v0, approx_matvec(A, v0))
show(figure_trace(trace_estimates, trace_true))

#%% Indeed, this even works with arbitrary matrix functions! Just have to apply f onto the Rayleigh-Ritz values
trace_true = np.sum(sf(ew))
trace_estimates = np.zeros(200)
for i in range(200):
  v0 = np.random.choice([-1, +1], size=A.shape[1])
  trace_estimates[i] = np.dot(v0, approx_matvec(A, v0, f=sf))  
show(figure_trace(trace_estimates, trace_true))

# %% Try xtrace! 
class Matvec:
  def __init__(self, A, f: Callable):
    self.A = A
    self.f = f
    self.shape = A.shape
  
  def matvec(self, v):
    return approx_matvec(A, v, f=self.f)

tr_est = xtrace(A)[0]
A.trace()

## Holy crap it seems to work with XTrace
from scipy.sparse.linalg import LinearOperator, aslinearoperator
L = aslinearoperator(Matvec(A, sf))
tr_est = xtrace(L)[0]
np.sum(sf(ew))

# %% Let's save all the quadratures nodes and weights so we can apply arbitrary matrix function
class LanczosQuadrature:
  def __init__(A, distribution: str = "rademacher"):
    pass
  
  def __call__(f: Callable = None):
    pass

# %% Can we just apply KDE to the Rayleigh-Ritz values? 





## Matrix function 
ew, U = np.linalg.eigh(A)

t0 = (U @ np.diag(f(ew)) @ U.T) @ v0
np.linalg.norm(y - t0)
z0 = Q @ V @ np.diag(rw) @ V.T @ Q.T @ v0 

# def softsign(x: float, q: int) -> float:
#   val = 0
#   for i in range(q+1):
#     if i > 0:
#       c = np.prod([(2*j - 1)/(2*j) for j in range(1, i+1)])
#       val += x * (1 - x**2)**i * c
#     else: 
#       val += x * (1 - x**2)**i
#   return val

import bokeh
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

## Caps out around 10k? But is very continuous 
p = figure(width=250, height=250)
s = soft_sign(10000)(np.linspace(-1, 1, 101))
p.line(np.linspace(-1, 1, 101), s)
show(p)







# %% Trace estimation by approximating the matrix function
sf = lambda ew: np.exp(-1.25*ew)
v0 = rademacher(A.shape[1])
(a,b), Q = lanczos(A, v0, max_steps=A.shape[1], orth=0, return_basis=True)
rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
# rw = rw / (rw + 1e-4)
rw = sf(rw)
y = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))

## Test the accuracy of Lanczos-FA on a matrix-function 
ew, ev = np.linalg.eig(A)
y_true = (ev @ np.diag(sf(ew)) @ ev.T) @ v0
mv_error = np.linalg.norm(y - y_true)
print(f"|| f(A)v - LanczosFA(A,v) ||_2 error = {mv_error:.8f}")
assert mv_error <= 1e-3 




y = trace_est(A, sf)[1]
# tr_est_cum = np.cumsum(trace_estimates) / (np.arange(len(trace_estimates))+1)
# tr_est_cum - trace_true

trace_est(A, sf)






## GET-MOMENTS Algorithm from Chen 
mom = Q.T @ v0
