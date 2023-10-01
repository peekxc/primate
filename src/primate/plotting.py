import numpy as np

import bokeh 
from bokeh.models import Span, Scatter, LinearAxis, Range1d, BoxAnnotation, Legend, Band, ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.models import NumeralTickFormatter
from scipy.special import erfinv

def figure_trace(info: dict, real_trace: float = None, **kwargs) -> figure:
  """Plots the trace estimates """

  samples = np.ravel(info['convergence']['samples'])
  sample_index = np.arange(1, len(samples)+1)
  min_samples = info['convergence']['min_num_samples']
  sample_avgs = np.cumsum(samples)/sample_index
  lanczos_degree = info['solver']['lanczos_degree']
  lanczos_orthogonalize = info['solver']['orthogonalize']

  ## uncertainty estimation (todo)
  quantile = np.sqrt(2) * erfinv(0.95)
  std_dev = np.nanstd(samples) 
  cumulative_abs_error = quantile * std_dev / np.sqrt(sample_index)
  cumulative_rel_error = (cumulative_abs_error / sample_avgs)

  p = figure(width=450, height=300, title=f"Stochastic trace estimates (degree={lanczos_degree}, orth={lanczos_orthogonalize})", **kwargs)
  p.toolbar_location = None
  p.scatter(sample_index, samples, size=4.0, color="gray", legend_label="samples")
  p.legend.location = "top_left"
  p.yaxis.axis_label = "Trace estimates"
  p.xaxis.axis_label = "Sample index"
  if (real_trace is not None):
    true_sp = Span(location=real_trace, dimension = "width", line_dash = "solid", line_color='red', line_width=1.0)
    p.add_layout(true_sp)
  p.line(sample_index, sample_avgs, line_color="black", line_width = 2.0, legend_label="mean estimate")

  ## Add confidence band
  band_source = ColumnDataSource(dict(x = sample_index, lower = sample_avgs - cumulative_abs_error, upper=sample_avgs + cumulative_abs_error))
  conf_band = Band(base="x", lower="lower", upper="upper", source=band_source, fill_alpha=0.3, fill_color="yellow", line_color="black")
  p.add_layout(conf_band)

  ## Error plot
  error_title = f"Error (converged: {np.take(info['convergence']['converged'], 0)})"
  q = figure(width=400, height=300, title=error_title, y_axis_location="left")
  q.toolbar_location = None
  q.yaxis.axis_label = "relative error"
  q.xaxis.axis_label = "Sample index"
  q.yaxis.formatter = NumeralTickFormatter(format='0%')
  q.y_range = Range1d(0, np.ceil(max(cumulative_rel_error)*100)/100, bounds = (0, 1))
  q.x_range = Range1d(0, len(sample_index))
  q.add_layout(BoxAnnotation(top=100, bottom=0, left=0, right=min_samples, fill_alpha=0.4, fill_color='#d3d3d3'))

  ## Plot the relative error
  rel_error_line = q.line(sample_index, cumulative_rel_error, line_width=2.5, line_color="gray")

  ## Plot the absolute error + its range
  q.extra_y_ranges = {"abs_error_rng": Range1d(start=0, end=np.ceil(max(cumulative_abs_error)))}
  q.add_layout(LinearAxis(y_range_name="abs_error_rng"), 'right')
  q.yaxis[1].axis_label = "absolute error"
  abs_error_line = q.line(sample_index, cumulative_abs_error, line_color="black", y_range_name="abs_error_rng")

  ## Shwo the thresholds for convergence
  abs_error = np.take(info['error']['absolute_error'], 0)
  rel_error = np.take(info['error']['relative_error'], 0)
  abs_error_threshold = q.line(x=[0, sample_index[-1]], y=[abs_error, abs_error], line_dash = "dashed", line_color='darkgray', line_width=1.0,  y_range_name="abs_error_rng")
  rel_error_threshold = q.line(x=[0, sample_index[-1]], y=[rel_error, rel_error], line_dash = "dotted", line_color='gray', line_width=1.0)

  legend_items = [('error (abs)', [abs_error_line]), ('error (rel)', [rel_error_line])]
  # legend_items += [('abs threshold', [abs_error_threshold])] + [('rel threshold', [rel_error_threshold])]
  legend = Legend(items=legend_items, location="top_right", orientation="horizontal", border_line_color="black")
  legend.label_standoff = 1
  legend.label_text_font_size = '10px'
  legend.padding = 2
  legend.spacing = 5
  q.add_layout(legend, "center")
  return row([p,q])
    
