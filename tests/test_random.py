import numpy as np
from typing import * 
from primate import random
from primate.random import _engines, _engine_prefixes, _random_gen

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

def test_seeding():
  s1 = random.rademacher(250, seed = -1)
  s2 = random.rademacher(250, seed = -1)
  assert any(s1 != s2), "random device not working"
  for i in np.random.choice(range(1000), size=10):
    s1 = random.rademacher(250, seed = i)
    s2 = random.rademacher(250, seed = i)
    assert all(s1 == s2), "seeding doesnt work"

def test_rademacher():
  assert np.all([r in [-1.0, +1.0] for r in random.rademacher(100)]), "Basic rademacher test failed"
  assert np.all(~np.isnan(random.rademacher(1500, rng='sx')))
  for rng in _engine_prefixes:
    c = random.rademacher(100*1500, rng=rng, seed=-1)
    counts = np.add.reduceat(c, np.arange(0, 100*1500, 100))
    cum_counts = np.cumsum(counts) / np.arange(1, len(counts)+1)
    assert abs(cum_counts[-1]) <= 1.0, f"Rademacher random number generator biased more than 1% (for engine {rng})"

def test_normal():
  assert np.all(~np.isnan(random.normal(1500, rng="sx")))
  for rng in _engine_prefixes:
    c = random.normal(100*1500, rng=rng, seed=-1)
    counts = np.add.reduceat(c, np.arange(0, 100*1500, 100))
    cum_counts = np.cumsum(counts) / np.arange(1, len(counts)+1)
    np.sum(random.normal(1500, rng=rng))
    assert abs(cum_counts[-1]) <= 1.0, f"Normal random number generator biased more than 1% (for engine {engine})"

# 2.69
# import timeit
# timeit.timeit(lambda: random.rademacher(5000), number=10000)

# def test_rayleigh():
#   for engine in _engine_prefixes:
#     averages = np.array([np.mean(random.rayleigh(100, engine=engine)) for _ in range(500)])
#     # assert np.allclose(sums, 1.0, atol = 1e-1), "Rayleigh vectors not normalized"
#     cum_avgs = np.cumsum(averages) / np.arange(1, len(averages)+1)
#     assert abs(cum_avgs[-1]) <= 1.0, "Rayleigh random number generator biased more than 1%"


## For interactive verification only 
# def plot_estimates():
#   from bokeh.io import output_notebook
#   from bokeh.plotting import show, figure
#   from bokeh.models import Span
#   output_notebook()
#   counts = np.array([np.sum(random.rademacher(100)) for _ in range(5000)])
#   # counts = np.array([np.sum(random.normal(100, engine="lcg")) for _ in range(5000)])
#   # counts = np.array([np.sum(random.rayleigh(100, engine="lcg")) for _ in range(5000)])
#   p = figure(width=200, height=150)
#   p.vbar(x=np.arange(len(counts)), top=counts, color="navy")
#   p.line(np.arange(1, len(counts)+1), np.cumsum(counts) / np.arange(1, len(counts)+1), color='red')
#   show(p)
