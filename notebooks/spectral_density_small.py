# %% 
import numpy as np
from primate.functional import spectral_density
from primate.random import symmetric
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
output_notebook()

## trace == 1
A = symmetric(n=5, ew=[0, 0.1, 0.2, 0.3, 0.4], pd=False)

## Q1: can we infer the nullspace from a degree-r Krylov expansion?
(dens, bins), info = spectral_density(A, bw=0.01, plot=True, rtol=0.0001, deg=3, orth=0)

## Q2: Is the error associated with the Krylov expansion 'uniform' in any sense?
density = np.zeros(100)
for i in range(1500):
  (dens, bins), info = spectral_density(A, bw=0.01, plot=False, deg=3, rtol=0.0001)
  density += dens

## Q2a: given a density that 'looks nice', does the area under the curve line up with the rank?
np.sum(dens[bins <= 0.05]) / np.sum(dens) # about 20.9%
np.sum(dens[bins > 0.05]) / np.sum(dens)  # about 80%

## Ans: while the density does seem to converge to something, its not centered at the eigenvalues
p = figure(width=350, height=150)
p.line(bins, density)
show(p)

# np.linalg.eigh(A)[0]
