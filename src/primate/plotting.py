import numpy as np
from typing import Union

def figure_trace(samples: Union[np.ndarray, dict], real_trace: float = None, **kwargs):
  """Plots the trace estimates """
  import bokeh 
  from bokeh.models import Span, Scatter, LinearAxis, Range1d, BoxAnnotation, Legend, Band, ColumnDataSource
  from bokeh.plotting import figure
  from bokeh.layouts import row, column
  from bokeh.models import NumeralTickFormatter
  from scipy.special import erfinv

  main_title = "Stochastic trace estimates"
  extra_titles = []
  if isinstance(samples, dict):
    extra_titles = []
    lanczos_degree = samples['lanczos_kwargs'].get('deg', np.nan)
    lanczos_orthogonalize = samples['lanczos_kwargs'].get('orth', np.nan) 
    extra_titles += [""] if np.isnan(lanczos_degree) else [f"deg={lanczos_degree}"]
    extra_titles += [""] if np.isnan(lanczos_orthogonalize) else [f"orth={lanczos_orthogonalize}"]

  ## Extract samples and take averages
  sample_vals = np.ravel(samples['samples']) if isinstance(samples, dict) else samples 
  valid_samples = sample_vals != 0
  n_samples = sum(valid_samples)
  sample_index = np.arange(1, n_samples+1)
  sample_avgs = np.cumsum(sample_vals[valid_samples])/sample_index
  main_title += ' (' + ', '.join(extra_titles) + ')' if len(extra_titles) > 0 else ''

  ## Uncertainty estimation
  conf = samples.get('confidence', 0.95) if isinstance(samples, dict) else 0.95
  quantile = np.sqrt(2.0)*erfinv(samples.get('confidence', 0.95)) if isinstance(samples, dict) else 1.959963984540054 
  std_dev = np.std(sample_vals[valid_samples], ddof=1) 
  std_error = std_dev / np.sqrt(sample_index)
  cum_abs_error = quantile * std_error            # CI margin of error 
  cum_rel_error = np.abs(std_error / sample_avgs) # coefficient of variation 

  ## Build the figure
  fig_title = "Stochastic trace estimates"
  if isinstance(samples, dict):
    fig_title += f" (degree={lanczos_degree}, orth={lanczos_orthogonalize})"
  p = figure(width=400, height=300, title=fig_title, **kwargs)
  p.toolbar_location = None
  p.scatter(sample_index, sample_vals, size=4.0, color="gray", legend_label="samples")
  p.legend.location = "top_left"
  p.yaxis.axis_label = f"Trace estimates ({(conf*100):.0f}% CI band)"
  p.xaxis.axis_label = "Sample index"
  if (real_trace is not None):
    true_sp = Span(location=real_trace, dimension = "width", line_dash = "solid", line_color='red', line_width=1.0)
    p.add_layout(true_sp)
  p.line(sample_index, sample_avgs, line_color="black", line_width = 2.0, legend_label="mean estimate")

  ## Add confidence band
  band_source = ColumnDataSource(dict(x = sample_index, lower = sample_avgs - cum_abs_error, upper=sample_avgs + cum_abs_error))
  conf_band = Band(base="x", lower="lower", upper="upper", source=band_source, fill_alpha=0.3, fill_color="yellow", line_color="black")
  p.add_layout(conf_band)

  ## Error plot
  error_title = "Estimator accuracy" 
  # if isinstance(samples, dict):
  #   error_title += f" (converged: {np.take(samples['convergence']['converged'], 0)})"
  q1 = figure(width=300, height=150, y_axis_location="left", title = error_title)
  q2 = figure(width=300, height=150, y_axis_location="left")
  q1.toolbar_location = None
  q2.toolbar_location = None
  q1.yaxis.axis_label = "Rel. std-dev (CV)"
  q2.yaxis.axis_label = f"Abs. error ({(conf*100):.0f}% CI)"
  q2.xaxis.axis_label = "Sample index"
  q1.yaxis.formatter = NumeralTickFormatter(format='0%')
  q1.y_range = Range1d(0, np.ceil(max(cum_rel_error)*100)/100, bounds = (0, 1))
  q2.x_range = q1.x_range = Range1d(0, len(sample_index))
  # if isinstance(samples, dict):
  #   q1.add_layout(BoxAnnotation(top=100, bottom=0, left=0, right=min_samples, fill_alpha=0.4, fill_color='#d3d3d3'))
  #   q2.add_layout(BoxAnnotation(top=100, bottom=0, left=0, right=min_samples, fill_alpha=0.4, fill_color='#d3d3d3'))

  ## Plot the relative error
  rel_error_line = q1.line(sample_index, cum_rel_error, line_width=2.5, line_color="gray")

  ## Plot the absolute error + its range
  # q.extra_y_ranges = {"abs_error_rng": Range1d(start=0, end=np.ceil(max(cumulative_abs_error)))}
  # q.add_layout(LinearAxis(y_range_name="abs_error_rng"), 'right')
  # q.yaxis[1].axis_label = "absolute error"
  abs_error_line = q2.line(sample_index, cum_abs_error, line_color="black")

  ## Show the thresholds for convergence towards the thresholds
  if isinstance(samples, dict):
    rel_error = np.take(samples['rtol'], 0)
    abs_error = np.take(samples['atol'], 0)
    rel_error_threshold = q1.line(x=[0, sample_index[-1]], y=[rel_error, rel_error], line_dash = "dotted", line_color='gray', line_width=1.0)
    abs_error_threshold = q2.line(x=[0, sample_index[-1]], y=[abs_error, abs_error], line_dash = "dashed", line_color='darkgray', line_width=1.0)
  
  ## Add the legend
  legend_items = [('atol', [abs_error_line]), ('rtol', [rel_error_line])]
  # legend_items += [('abs threshold', [abs_error_threshold])] + [('rel threshold', [rel_error_threshold])]
  legend = Legend(items=legend_items, location="top_right", orientation="horizontal", border_line_color="black")
  legend.label_standoff = 1
  legend.label_text_font_size = '10px'
  legend.padding = 2
  legend.spacing = 5
  
  # q1.add_layout(legend, "center")
  return row([p,column([q1,q2])])
    
def plot_trace(info: dict, real_trace: float = None, **kwargs) -> None:
  from bokeh.plotting import show
  show(figure_trace(info, real_trace, **kwargs))