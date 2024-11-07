import numpy as np
from bokeh.plotting import figure


def figure_csm(values: np.ndarray):
	m = 1 / len(values)
	csm = lambda x: np.searchsorted(values, x) * m
	p = figure(
		width=350,
		height=250,
		title="Cumulative spectral density",
		x_axis_label="Spectrum",
		y_axis_label=r"$$\mathbf{1}(\lambda \leq x)$$",
	)
	p.title.align = "center"
	p.line(values, csm(values))
	p.scatter(values, 0, size=7.5, color="red", marker="x", legend_label="Eigenvalues")
	p.varea_step(x=np.append(values, 1.0), y1=np.zeros(len(values) + 1), y2=np.append(csm(values), 1.0), fill_alpha=0.15)
	p.legend.location = "top_left"
	p.legend.margin = 5
	p.legend.padding = 2
	p.toolbar_location = None
	return p
