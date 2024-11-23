import numpy as np
import scipy as sp
from typing import Optional


def figure_csm(values: np.ndarray, **kwargs):
	from bokeh.plotting import figure

	m = 1 / len(values)
	csm = lambda x: np.searchsorted(values, x) * m
	p = figure(
		width=350,
		height=250,
		title="Cumulative spectral density",
		x_axis_label="Spectrum",
		y_axis_label=r"$$\mathbf{1}(\lambda \leq x)$$",
		**kwargs,
	)
	p.title.align = "center"
	# x = np.linspace(np.min(values), np.max(values), 5000)
	# p.line(x, csm(x))
	p.scatter(values, 0, size=7.5, color="red", marker="x", legend_label="Eigenvalues")
	p.varea_step(x=np.append(values, 1.0), y1=np.zeros(len(values) + 1), y2=np.append(csm(values), 1.0), fill_alpha=0.15)
	p.legend.location = "top_left"
	p.legend.margin = 5
	p.legend.padding = 2
	p.toolbar_location = None
	return p


def figure_orth_poly():
	pass


def figure_jacobi(deg: int = 4, alpha: float = 0, beta: float = 0):
	assert deg <= 10
	from bokeh.plotting import figure
	from bokeh.palettes import Category10

	colors = Category10[deg]
	title = "\\text{Jacobi polynomials }"
	title += f"(\\alpha={alpha:.1f}, \\beta={beta:.1f})"
	p = figure(width=350, height=300, title="$$" + title + "$$")
	p.title.text_font_style = "normal"
	p.title.align = "center"
	dom = np.linspace(-1, 1, 1500)
	for d, col in zip(range(deg), colors):
		y = sp.special.eval_jacobi(d, alpha, beta, dom)
		p.line(dom, y, line_color=col, line_width=1.5, legend_label=f"d={d}")
	p.legend.location = "bottom_right"
	# p.legend.title = "Degree"
	p.legend.padding = 4
	p.legend.spacing = -10
	p.legend.margin = 4
	p.legend.title_text_font_size = "10px"
	p.legend.label_text_font_size = "10px"
	p.legend.label_height = 0
	# p.legend.orientation = "horizontal"
	return p


def figure_sequence(samples: np.ndarray, mu: Optional[float] = None, **kwargs: dict):
	"""Generates figures showing the convergence of sample estimates."""
	from bokeh.models import Span
	from bokeh.plotting import figure

	# ## Extract samples and take averages
	sample_vals = np.ravel(samples)
	valid_samples = sample_vals != 0
	n_samples = np.sum(valid_samples)
	sample_index = np.arange(1, n_samples + 1)
	sample_avgs = np.cumsum(sample_vals[valid_samples]) / sample_index

	## Build the figure
	fig_title = "Monte Carlo sample variates"
	p = figure(width=400, height=300, title=fig_title, **kwargs)
	p.toolbar_location = None
	p.scatter(sample_index, sample_vals, size=4.0, color="gray", legend_label="samples")
	p.title.align = "center"
	p.legend.location = "top_left"
	p.yaxis.axis_label = "Estimates"
	p.xaxis.axis_label = "Sample index"
	if mu is not None:
		true_sp = Span(location=mu, dimension="width", line_dash="solid", line_color="red", line_width=1.0)
		p.add_layout(true_sp)
	p.line(sample_index, sample_avgs, line_color="black", line_width=1.5, legend_label="estimator")  # line_dash="dotted"

	p.legend.padding = 4
	p.legend.spacing = 2
	p.legend.margin = 4
	p.legend.title_text_font_size = "11px"
	p.legend.label_text_font_size = "11px"
	return p


# def add_confidence_band(p):
# 	from bokeh.models import Band, ColumnDataSource, Legend, NumeralTickFormatter, Range1d, Span
# 	# ## Uncertainty estimation
# 	# quantile = np.sqrt(2.0) * sp.special.erfinv(self.confidence)
# 	# std_dev = np.std(sample_vals[valid_samples], ddof=1)
# 	# std_error = std_dev / np.sqrt(sample_index)
# 	# cum_abs_error = quantile * std_error  # CI margin of error
# 	# cum_rel_error = np.abs(std_error / sample_avgs)  # coefficient of variation

# 	## Add confidence band
# 	band_source = ColumnDataSource(
# 		dict(x=sample_index, lower=sample_avgs - cum_abs_error, upper=sample_avgs + cum_abs_error)
# 	)
# 	conf_band = Band(
# 		base="x", lower="lower", upper="upper", source=band_source, fill_alpha=0.3, fill_color="yellow", line_color="black"
# 	)
# 	p.add_layout(conf_band)


# def figure_est_error(results: EstimatorResult, absolute: bool = True, title: str = "Estimator accuracy"):
# 	if absolute:
# 		q2 = figure(width=300, height=150, y_axis_location="left")
# 		q2.toolbar_location = None
# 		q2.yaxis.axis_label = f"Abs. error ({(self.confidence*100):.0f}% CI)"
# 		q2.xaxis.axis_label = "Sample index"
# 		q2.x_range = Range1d(0, len(sample_index))

# 		## Plot the absolute error + thresholds for convergence
# 		abs_error_line = q2.line(sample_index, cum_abs_error, line_color="black")
# 		abs_error_threshold = q2.line(
# 			x=[0, sample_index[-1]], y=[self.atol, self.atol], line_dash="dashed", line_color="darkgray", line_width=1.0
# 		)
# 		return q2
# 	else:
# 		q1 = figure(width=300, height=150, y_axis_location="left", title=title)
# 		q1.toolbar_location = None
# 		q1.yaxis.axis_label = "Rel. std-dev (CV)"
# 		q1.yaxis.formatter = NumeralTickFormatter(format="0.00%")
# 		q2.x_range = Range1d(0, len(sample_index))
# 		q1.y_range = Range1d(0, np.ceil(max(cum_rel_error) * 100) / 100, bounds=(0, 1))

# 		## Plot the relative error + thresholds for convergence
# 		rel_error_line = q1.line(sample_index, cum_rel_error, line_width=1.5, line_color="gray")
# 		rel_error_threshold = q1.line(
# 			x=[0, sample_index[-1]], y=[self.rtol, self.rtol], line_dash="dotted", line_color="gray", line_width=1.0
# 		)
# 		return q1


# ## Add the legend
# legend_items = [("atol", [abs_error_line]), ("rtol", [rel_error_line])]
# # legend_items += [('abs threshold', [abs_error_threshold])] + [('rel threshold', [rel_error_threshold])]
# legend = Legend(items=legend_items, location="top_right", orientation="horizontal", border_line_color="black")
# legend.label_standoff = 1
# legend.label_text_font_size = "10px"
# legend.padding = 2
# legend.spacing = 5

# q1.add_layout(legend, "center")
# return row([p, column([q1, q2])])


# _builtin_matrix_functions = ["identity", "abs", "sqrt", "log", "inv", "exp", "smoothstep", "numrank", "gaussian"]
# def figure_fun(fun: Union[str, Callable], bounds: tuple = (-1, 1), *args, **kwargs):
# 	assert isinstance(fun, str) or isinstance(fun, Callable), "'fun' must be string or callable."
# 	from bokeh.plotting import figure

# 	dom = np.linspace(bounds[0], bounds[1], 250, endpoint=True)
# 	p = figure(width=250, height=250, title=f"fun = {fun}")
# 	fun = param_callable(fun, **kwargs) if isinstance(fun, str) else fun
# 	out = fun(dom)
# 	p.line(dom, out)
# 	return p
