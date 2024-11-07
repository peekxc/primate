import numpy as np


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
	x = np.linspace(np.min(values), np.max(values), 5000)
	p.line(x, csm(x))
	p.scatter(values, 0, size=7.5, color="red", marker="x", legend_label="Eigenvalues")
	p.varea_step(x=np.append(values, 1.0), y1=np.zeros(len(values) + 1), y2=np.append(csm(values), 1.0), fill_alpha=0.15)
	p.legend.location = "top_left"
	p.legend.margin = 5
	p.legend.padding = 2
	p.toolbar_location = None
	return p


# def figure_cesm():


def figure_orth_poly():
	pass
