# %% Imports
import json
import subprocess
import timeit
from collections import namedtuple
from tempfile import NamedTemporaryFile
from typing import Callable

from geomcover.io import sparray
import numpy as np
from bokeh.io import output_notebook
from bokeh.layouts import column, row
from bokeh.plotting import figure, show
from bokeh.models import Legend, NumeralTickFormatter, BasicTickFormatter, CustomJSTickFormatter
from geomcover.csgraph import cycle_graph
from map2color import map2hex
from memray import Tracker
from primate.operators import MatrixFunction, Toeplitz
from primate.stochastic import symmetric
from primate.special import param_callable
from scipy.linalg import toeplitz
from scipy.sparse import diags, sparray
from scipy.sparse.linalg import eigsh, LinearOperator
from primate.operators import normalize_unit


output_notebook()


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


# %%
def dac_matrix_function(A, fun, x):
	ew, ev = np.linalg.eigh(A)
	return ev @ np.diag(fun(ew)) @ ev.T @ x


def arpack_matrix_function(A, fun, x):
	ew, ev = eigsh(A, k=(A.shape[0] - 1), which="LM", v0=x)
	# np.linalg.norm(x) * ev @ ((ew * ev[0, :] ** 2)[:, np.newaxis]).ravel()
	# v_norm * ev @ (self._fun(rw) * Y[0, :])[:, np.newaxis]
	# return ev @ diags(fun(ew)) @ ev.T @ x
	return (ev * fun(ew)) @ ev.T @ x


## Get the true eigenvalues (does this matter?)
# true_ew = np.sort(1.0 - np.cos(2 * np.pi * np.arange(n) / n))
# z = L @ x
# %%
def benchmark_matrix_func(
	A_gen: Callable[int, sparray], fun: Callable, sizes: list = [10, 50, 100, 150, 200, 250], repeat: int = 15
):
	benchmark = namedtuple("Benchmark", "name size timings memory error degree")
	rng = np.random.default_rng(1234)
	results = []
	mat_sizes = np.array(sizes)
	for n in mat_sizes:
		A_op = A_gen(n)
		A_dense = A_op @ np.eye(A_op.shape[0])
		x = rng.uniform(size=n, low=-1.0, high=1.0)

		## Divide and conquer
		z_dac = np.ravel(dac_matrix_function(A_dense, fun, x))
		timings = timeit.repeat(lambda: dac_matrix_function(A_dense, fun, x), repeat=repeat, number=1)
		memory = peak_memory(dac_matrix_function, A=A_dense, fun=fun, x=x)
		bench = benchmark("dac", n, timings, memory["metadata"]["peak_memory"], 0.0, n)
		results.append(bench)

		## ARPACK
		z_arp = arpack_matrix_function(A_op, fun, x)
		timings = timeit.repeat(lambda: arpack_matrix_function(A_op, fun, x), repeat=repeat, number=1)
		memory = peak_memory(arpack_matrix_function, A=A_op, fun=fun, x=x)
		bench = benchmark(
			"arp", n, timings, memory["metadata"]["peak_memory"], np.linalg.norm(z_dac - z_arp) / np.linalg.norm(z_dac), n
		)
		results.append(bench)

		## Lanczos
		N = list(map(int, [3, max(3, np.ceil(np.log(n))), np.ceil(np.sqrt(n)), n // 4, n // 2, n]))
		for ii, deg in enumerate(N):
			M = MatrixFunction(A_op, fun=fun, deg=deg, orth=0)
			z_lan = M @ x
			timings = timeit.repeat(lambda: M @ x, repeat=repeat, number=1)
			memory = peak_memory(lambda: M @ x)
			bench = benchmark(
				"lan", n, timings, memory["metadata"]["peak_memory"], np.linalg.norm(z_dac - z_lan) / np.linalg.norm(z_dac), ii
			)
			results.append(bench)
		print(n)
	return results


# Plot the timings, the maximum memory used, and the perform across multiple degrees, the accuracies
def plot_agg_results(agg_results, aux_title="", add_legend: bool = False):
	d_kw = dict(toolbar_location=None, x_axis_label="Number of vertices (n)")
	width, height = 350, 275

	p = figure(
		width=width,
		height=height,
		title="Matvec runtime" + aux_title,
		y_axis_type="log",
		y_axis_label="Seconds (log-scale)",
		**d_kw,
	)
	q = figure(width=width, height=height, title="Peak memory usage" + aux_title, **d_kw)  # y_axis_label="Bytes"
	r = figure(
		width=width + 75 * int(add_legend),
		height=height,
		title="Approximation error" + aux_title,
		y_axis_type="log",
		y_axis_label="Mean residual norm",
		**d_kw,
	)
	p.title.align = q.title.align = r.title.align = "center"

	p.yaxis.major_label_orientation = 3.14159 / 6
	q.yaxis.major_label_orientation = 3.14159 / 6
	r.yaxis.major_label_orientation = 3.14159 / 6

	legends = []

	q.yaxis[0].formatter = BasicTickFormatter(precision=0, use_scientific=True)
	r.yaxis[0].formatter = BasicTickFormatter(precision=0, use_scientific=True)

	q.yaxis.formatter = NumeralTickFormatter(format="0 b")

	p.yaxis.axis_label_standoff = 0
	q.yaxis.axis_label_standoff = 0
	r.yaxis.axis_label_standoff = 0
	sca_ops = dict(size=8, line_color="darkgray", line_width=1.5)
	lin_ops = dict(line_width=2.25)

	mat_sizes = np.unique(agg_results.f1)
	mop_time = agg_results[agg_results.f0 == "dac"].f2
	mop_memory = agg_results[agg_results.f0 == "dac"].f3
	p.line(mat_sizes, mop_time, line_color="blue", **lin_ops)
	p.scatter(mat_sizes, mop_time, color="blue", **sca_ops)
	q.line(mat_sizes, mop_memory, line_color="blue", **lin_ops)
	q.scatter(mat_sizes, mop_memory, color="blue", **sca_ops)
	dac_lr = r.line(mat_sizes, agg_results[agg_results.f0 == "dac"].f4, line_color="blue", **lin_ops)

	legends.append(("DAC", [dac_lr]))

	mop_time = agg_results[agg_results.f0 == "arp"].f2
	mop_memory = agg_results[agg_results.f0 == "arp"].f3
	p.line(mat_sizes, mop_time, line_color="red", **lin_ops)
	p.scatter(mat_sizes, mop_time, color="red", **sca_ops)
	q.line(mat_sizes, mop_memory, line_color="red", **lin_ops)
	q.scatter(mat_sizes, mop_memory, color="red", **sca_ops)

	irl_lr = r.line(mat_sizes, agg_results[agg_results.f0 == "arp"].f4, line_color="red", **lin_ops)
	r.scatter(mat_sizes, agg_results[agg_results.f0 == "arp"].f4, color="red", **sca_ops)
	legends.append(("IRC", [irl_lr]))

	orange_cols = map2hex([0, 1, 2, 3, 4, 5, 6], "YlOrRd")
	orange_cols = orange_cols[:-1]

	deg_labels = ["C=3", "log(n)", "sqrt(n)", "n/4", "n/2", "n"]
	for deg_i, col, deg_label in zip(range(6), orange_cols, deg_labels):
		sel = np.logical_and(agg_results.f0 == "lan", agg_results.f5 == deg_i)
		mop_time = agg_results[sel].f2
		mop_memory = agg_results[sel].f3
		lan_lr = p.line(mat_sizes, mop_time, line_color=col, line_dash="dashed", **lin_ops)
		p.scatter(mat_sizes, mop_time, color=col, **sca_ops)
		q.line(mat_sizes, mop_memory, line_color=col, line_dash="dashed", **lin_ops)
		q.scatter(mat_sizes, mop_memory, color=col, **sca_ops)
		lan_lr = r.line(mat_sizes, agg_results[sel].f4 / agg_results[sel].f1, line_color=col, line_dash="dashed", **lin_ops)
		r.scatter(mat_sizes, agg_results[sel].f4 / agg_results[sel].f1, color=col, **sca_ops)
		legends.append(("Lan " + deg_label, [lan_lr]))

	for fig in [p, q, r]:
		fig.title.text_font_size = "14pt"  # Larger title font
		fig.xaxis.axis_label_text_font_size = "13pt"  # Larger x-axis label font
		fig.yaxis.axis_label_text_font_size = "13pt"  # Larger y-axis label font
		fig.xaxis.major_label_text_font_size = "12pt"  # Larger x-axis tick label font
		fig.yaxis.major_label_text_font_size = "12pt"  # Larger y-axis tick label font
		# fig.legend.label_text_font_size = "12pt"  # Larger legend font

	if add_legend:
		legend = Legend(items=legends)
		legend.spacing = 0
		legend.padding = 5
		legend.margin = 5
		legend.label_text_font_size = "16px"
		# p.add_layout(legend, "right")
		# q.add_layout(legend, "right")
		r.add_layout(legend, "right")

	return row(p, q, r)


# %%
rng = np.random.default_rng(1234)
# fun = lambda x: np.exp(-10.0 * np.abs(x))
fun = lambda x: np.reciprocal(np.abs(x) + 1e-1)
# fun = lambda x: np.sqrt(np.maximum(x, 0))

circle_gen = lambda n: normalize_unit(cycle_laplacian(n))
toeplitz_gen = lambda n: normalize_unit(Toeplitz(rng.uniform(size=n, low=-1, high=1)))
symmetric_gen = lambda n: normalize_unit(symmetric(n))

sizes = [50, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000]
agg_results = {}
for gen, name in zip([circle_gen, toeplitz_gen, symmetric_gen], ["circle", "toeplitz", "symmetric"]):
	results = benchmark_matrix_func(gen, fun, sizes=sizes, repeat=1)
	agg_results[name] = np.rec.array([(r.name, r.size, np.min(r.timings), r.memory, r.error, r.degree) for r in results])


p1 = plot_agg_results(agg_results["circle"], " (Circle)", add_legend=True)
p2 = plot_agg_results(agg_results["toeplitz"], " (Toeplitz)", add_legend=True)
p3 = plot_agg_results(agg_results["symmetric"], " (PSD)", add_legend=True)

## post process
for p in p1.children:
	p.xaxis.axis_label = ""

for p in p2.children:
	p.xaxis.axis_label = ""
	# p.title.visible = False

# p1.children[2].axis[1]
# for p in p3.children:
# p.title.visible = False

show(column(p1, p2, p3))
# show(plot_agg_results(agg_results))
