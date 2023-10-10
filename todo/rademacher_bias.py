import numpy as np 
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.models import Span
output_notebook()

# from pyimate import rademacher
from pyimate import _random_generator

# %% Test openmp
z = np.zeros(100, np.float32)
_random_generator.rademacher_xs(z, 1)

# %% Test Rademacher
def rademacher(n: int):
  z = np.zeros(n, np.float32)
  _random_generator.rademacher_xs(z, 1)
  return z

counts = np.array([np.sum(rademacher(100)) for _ in range(5000)])
p = figure(width=200, height=150)
p.vbar(x=np.arange(len(counts)), top=counts, color="navy")
p.line(np.arange(1, len(counts)+1), np.cumsum(counts) / np.arange(1, len(counts)+1), color='red')
show(p)

  # _random_generator.rademacher_pcg_single(z)
# %% Rademacher (fast)
counts = np.array([np.sum(rademacher(100)) for _ in range(5000)])
p = figure(width=200, height=150)
p.vbar(x=np.arange(len(counts)), top=counts, color="navy")
p.line(np.arange(1, len(counts)+1), np.cumsum(counts) / np.arange(1, len(counts)+1), color='red')
show(p)

# %% Rademacher numpy
counts = np.array([np.sum(np.random.choice([-1.0, +1.0], size=100)) for _ in range(5000)])
p = figure(width=200, height=150)
p.vbar(x=np.arange(len(counts)), top=counts, color="navy")
p.line(np.arange(1, len(counts)+1), np.cumsum(counts) / np.arange(1, len(counts)+1), color='red')
show(p)

# %% Rademacher benchmarks -- results are mixed, but indeed they are different! they are being parallelized
from timeit import timeit
import numpy as np
from pyimate import _random_generator
z = np.zeros(1500000, np.float32)

timeit(lambda: _random_generator.rademacher_xs(z, 1), number=10000)
timeit(lambda: _random_generator.rademacher_xs(z, 2), number=10000)
timeit(lambda: _random_generator.rademacher_xs(z, 3), number=10000)
timeit(lambda: _random_generator.rademacher_xs(z, 4), number=10000)
timeit(lambda: _random_generator.rademacher_xs(z, 6), number=10000) # native 
timeit(lambda: _random_generator.rademacher_xs(z, 8), number=10000)
