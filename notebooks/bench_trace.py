# %% Imports
import json
import subprocess
import timeit
from collections import namedtuple
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
from bokeh.io import output_notebook
from bokeh.layouts import column, row
from bokeh.models import BasicTickFormatter, CustomJSTickFormatter, Legend, NumeralTickFormatter
from bokeh.plotting import figure, show
from geomcover.csgraph import cycle_graph
from map2color import map2hex
from memray import Tracker
from notebooks.sparsifier import spectral_sparsifier
from primate.lanczos import lanczos
from primate.estimators import hutch
from primate.operators import MatrixFunction, Toeplitz, normalize_unit
from primate.special import param_callable
from primate.stochastic import symmetric
from primate.quadrature import lanczos_quadrature
from scipy.linalg import toeplitz
from scipy.sparse import diags, sparray
from scipy.sparse.linalg import eigsh

from primate.tridiag import eigvalsh_tridiag

output_notebook()


# %%
def cycle_laplacian(n: int, normalized: bool = True):
	A = cycle_graph(n, 2)
	if normalized:
		L = diags(np.ones(n)) - (1 / 2) * A
	else:
		d = A @ np.ones(A.shape[0])
		L = diags(d) - A
	return L


def peak_memory(func: Callable, *args, **kwargs):
	temp_file = NamedTemporaryFile()
	with Tracker(temp_file.name + ".bin"):
		func(*args, **kwargs)
	tfn = temp_file.name
	subprocess.run(f"memray stats --json -o {tfn}.json {tfn}.bin", shell=True, capture_output=True, check=True)
	return json.load(open(f"{tfn}.json", "r"))


# %% functions to benchmark
def dac_trace(A, fun):
	ew, ev = np.linalg.eigh(A)
	# return np.trace(ev @ np.diag(fun(ew)) @ ev.T)
	return np.sum(fun(ew))


def arpack_trace(A, fun):
	ew = eigsh(A, k=(A.shape[0] - 1), which="LM", return_eigenvectors=False)
	return np.sum(fun(ew))


def lanczos_trace(A, fun, **kwargs):
	a, b = lanczos(A, **kwargs)
	nodes, weights = lanczos_quadrature(a, b, **kwargs)
	return np.sum(fun(nodes) * weights)


# %%
def benchmark_trace(
	name: str,
	A_gen: Callable[int, sparray],
	fun: Callable,
	bench_fun: Callable,
	dense: bool = False,
	sizes: list = [10, 50, 100, 150, 200, 250],
	repeat: int = 15,
	**kwargs,
):
	benchmark = namedtuple("Benchmark", "name size timings memory output")
	results = []
	mat_sizes = np.array(sizes)
	for n in mat_sizes:
		A_op = A_gen(n)
		A_op = A_op if not dense else A_op @ np.eye(A_op.shape[0])
		out = np.ravel(bench_fun(A_op, fun, **kwargs))
		timings = timeit.repeat(lambda: bench_fun(A_op, fun), repeat=repeat, number=1)
		memory = peak_memory(bench_fun, A=A_op, fun=fun)
		bench = benchmark(name, n, timings, memory["metadata"]["peak_memory"], out.item())
		results.append(bench)
	return results


# %%
from scipy.sparse import random_array


def random_sp(n: int):
	A = random_array((n, n), density=np.sqrt(n) / n)
	A = (A + A.T) / 2
	A.setdiag(0)
	A = A.tocsc()
	A.sort_indices()
	return A


# fun = lambda x: np.exp(-10.0 * np.abs(x))
# fun = lambda x: np.sqrt(np.maximum(x, 0))
fun = lambda x: np.reciprocal(np.abs(x) + 1e-1)

circle_gen = lambda n: normalize_unit(cycle_laplacian(n))
symmetric_gen = lambda n: normalize_unit(symmetric(n))
random_gen = lambda n: normalize_unit(random_sp(n))

# %%
sizes = [50, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000]
agg_results = {}
for gen, name in zip([circle_gen, random_gen, symmetric_gen], ["circle", "random", "symmetric"]):
	results = benchmark_trace(gen, fun, sizes=sizes, repeat=1)
	agg_results[name] = np.rec.array([(r.name, r.size, np.min(r.timings), r.memory, r.output) for r in results])

from scipy.sparse import diags

d = A @ np.ones(A.shape[0])
L = diags(d) - A


from primate.tridiag import eigvalsh_tridiag, eigh_tridiag

ew, _ = np.linalg.eigh(L.todense())
np.sum(ew)

# lanczos_trace(L, fun=lambda x: x, method="mrrr")
# a, b = lanczos(L, deg=L.shape[0] + 1, orth=L.shape[0])
# rw, rv = eigh_tridiag(a, b)
# np.sum(rw)
# L.shape[0] * np.sum(rw * np.square(rv[0, :]))


# L.shape[0] * np.sum(nodes * weights)
# np.mean(hutch(L, maxiter=15000, atol=0, rtol=0, seed=0))

# %%
# from primate.stats import MeanEstimatorCLT, ControlVariableEstimator, control_variate_estimator

# est = MeanEstimatorCLT()
# est(samples)

# ew = np.linalg.eigh(L.todense())[0]
# mu = L.trace()
# samples = hutch(L, fun=lambda x: x, seed=1234, deg=20, atol=None, rtol=None, maxiter=300)


# samples_f1 = hutch(L, fun=lambda x: np.exp(-0.10 * x), seed=1234, deg=3, atol=None, rtol=None, maxiter=300)
# samples_f2 = hutch(
# 	L, fun=lambda x: np.exp(-0.10 * x), seed=1234, deg=350, ncv=350, orth=350, atol=None, rtol=None, maxiter=300
# )

# np.mean(samples_f1)
# np.mean(samples_f2)

# show(MeanEstimatorCLT(samples_f1).plot(samples_f1))
# show(MeanEstimatorCLT(samples_f2).plot(samples_f2))

# np.abs(np.mean(samples_f) - mu_true) <= 0.0002 * mu_true

# control_variate_estimator(samples_f, cvs=samples, mu=mu)

## Relative + absolute error guarantee
# print(len(samples))
# assert abs(est.mu_est - mu) <= 0.01 * mu or abs(mu - est.mu_est) <= 0.5

# mu_true = L.trace()
# est = ControlVariableEstimator(f=np.mean, ev=mu_true)

# est(samples)

# samples = hutch(A, fun=lambda x: x / (x + 1e-1), seed=1234, stop=None, deg=20)


# show(est.plot(samples, L.trace()))
