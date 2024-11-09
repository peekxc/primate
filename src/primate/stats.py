from numbers import Number
from typing import Union, Optional, Callable

import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, t, sem


## See also:
## https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
## https://cran.r-project.org/web/packages/distributions3/vignettes/one-sample-t-confidence-interval.html
# Equivalent manual approach
# sq_n, ssize = np.sqrt(len(a)), (len(a)-1)
# s = np.std(a, ddof=1) # == (1.0 / np.sqrt(ssize)) * np.sum((a - mean)**2))
# rem = (1.0 - conf) / 2.0
# upper = st.t.ppf(1.0 - rem, ssize)
# lower = np.negative(upper)
# c_interval = mean + np.array([lower, upper]) * s / sq_n
# np.sqrt(2) * erfinv(2*0.025 - 1)
def confidence_interval(a: np.ndarray, confidence: float = 0.95, sdist: str = "t") -> tuple:
	"""Confidence intervals for the sample mean of a set of measurements."""
	assert isinstance(confidence, Number) and confidence >= 0.0 and confidence <= 1.0, "Invalid confidence measure"
	if sdist == "t":
		mean, std_err, m = np.mean(a), sem(a, ddof=1), t.ppf((1 + confidence) / 2.0, len(a) - 1)
		return mean - m * std_err, mean + m * std_err
	elif sdist == "normal":
		sq_n = np.sqrt(len(a))
		mean, std = np.mean(a), np.std(a, ddof=1)
		return norm.interval(confidence, loc=mean, scale=std / sq_n)
	else:
		raise ValueError(f"Unknown sampling distribution '{sdist}'.")


# class MaxiterCounter:
# 	def __init__(self, maxiter: int = 200) -> None:
# 		self.it = 0
# 		self.maxiter = maxiter

# 	def __call__(self, estimates: Union[float, np.ndarray, None] = None) -> bool:
# 		self.it += 1
# 		return self.it > self.maxiter


class ControlVariateEstimator:
	def __init__(self, f: Callable[np.ndarray, float], ev: float, alpha: Optional[float] = None):
		self.ev = ev
		self.f = f
		self.alpha = alpha

	def __call__(self, samples: np.ndarray):
		n = len(samples)
		cvs = self.f(samples)
		if self.alpha is None:
			C = np.cov(samples, cvs, ddof=1)  # sample covariance
			alpha = C[0, 1] / C[1, 1]
		else:
			alpha = self.alpha
		cv_est = np.mean(samples - alpha * (cvs - self.ev))
		SE = sem(samples)
		C_inner = (1 - C[0, 1] ** 2 / np.prod(np.diag(C))) * C[0, 0]
		SE = np.sqrt((1 / n) * C_inner)
		z = norm.ppf(1.0 - (alpha / 2))
		return cv_est, (cv_est - z * SE, cv_est - z * SE)


def control_variate_estimator(samples: np.ndarray, cvs: np.ndarray, mu: float, alpha: Optional[float] = None):
	assert len(samples) == len(cvs), "Number of control variables must match number of samples."
	n = len(samples)
	if alpha is None:
		C = np.cov(samples, cvs, ddof=1)  # sample covariance
		alpha = C[0, 1] / C[1, 1]
	denom = np.arange(n)
	denom[0] = 1
	cv_est = (samples - alpha * (cvs - mu)) / denom
	# SE = sem(samples)
	C_inner = (1 - C[0, 1] ** 2 / np.prod(np.diag(C))) * C[0, 0]
	SE = np.sqrt((1 / n) * C_inner)
	z = norm.ppf(1.0 - (alpha / 2))
	return cv_est, (cv_est[-1] - z * SE, cv_est[-1] - z * SE)


# See: https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
# def _parameterize_stop(criterion: str = "confidence") -> Callable:
class MeanEstimatorCLT:
	"""Parameterizes an expected value estimator that checks convergence of a sample mean within a confidence interval using the CLT.

	Provides the following methods:
		- __call__ = Updates the estimator with newly measured samples
		- converged = Checks convergence of the estimator within an interval
		-	plot = Plots the samples and their sample distribution CI's

	"""

	def __init__(self, samples: list = None, confidence: float = 0.95, atol: float = 0.05, rtol: float = 0.01) -> None:
		self.mu_est, self.vr_est = 0.0, 0.0
		self.mu_pre, self.vr_pre = 0.0, 0.0
		self.n_samples = 0
		self.z = 2 ** (1 / 2) * erfinv(confidence)
		self.t_scores = t.ppf((confidence + 1.0) / 2.0, df=np.arange(30) + 1)
		self.atol = 0.0 if atol is None else atol
		self.rtol = 0.0 if rtol is None else rtol
		self.margin_of_error = np.inf
		self.confidence = confidence
		if samples is not None:
			self.__call__(samples)

	## Bulk update function, which keeps a running mean and
	def __call__(self, estimates: Union[float, np.ndarray, None] = None) -> bool:
		if estimates is None:
			return self.converged()
		estimates = np.array([estimates]).ravel()
		for estimate in estimates:
			self.n_samples += 1
			denom = 1.0 / float(self.n_samples)
			L = float(self.n_samples - 2) / float(self.n_samples - 1) if self.n_samples > 2 else 0.0
			self.mu_est = denom * (estimate + (self.n_samples - 1) * self.mu_pre)
			self.mu_pre = self.mu_est if self.n_samples == 1 else self.mu_pre
			self.vr_est = L * self.vr_pre + denom * (estimate - self.mu_pre) ** 2  # update sample variance
			self.mu_pre = self.mu_est
			self.vr_pre = self.vr_est
		return self.converged()

	def converged(self) -> bool:
		if self.n_samples < 3:
			return False
		std_dev = self.vr_est ** (1 / 2)
		score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
		self.margin_of_error = score * std_dev / float(self.n_samples) ** (1 / 2)  # todo: remove sqrt's
		std_error = std_dev / np.sqrt(self.n_samples)
		rel_error = abs(std_error / self.mu_est)
		return self.margin_of_error <= self.atol or rel_error <= self.rtol

	def __repr__(self) -> str:
		moe = self.margin_of_error
		msg = f"Est: {self.mu_est:.3f} +/- {moe:.3f}"
		msg += f" ({self.confidence*100:.0f}% CI,"  # | {(cv*100):.0f}% CV
		msg += f" #S:{ self.n_samples })"
		return msg

	def plot(self, samples: np.ndarray, mu: Optional[float] = None, **kwargs: dict):
		"""Generates figures showing the convergence of sample estimates."""
		from bokeh.layouts import column, row
		from bokeh.models import Band, ColumnDataSource, Legend, NumeralTickFormatter, Range1d, Span
		from bokeh.plotting import figure

		## Extract samples and take averages
		sample_vals = np.ravel(samples)
		valid_samples = sample_vals != 0
		n_samples = sum(valid_samples)
		sample_index = np.arange(1, n_samples + 1)
		sample_avgs = np.cumsum(sample_vals[valid_samples]) / sample_index

		## Uncertainty estimation
		quantile = np.sqrt(2.0) * erfinv(self.confidence)
		std_dev = np.std(sample_vals[valid_samples], ddof=1)
		std_error = std_dev / np.sqrt(sample_index)
		cum_abs_error = quantile * std_error  # CI margin of error
		cum_rel_error = np.abs(std_error / sample_avgs)  # coefficient of variation

		## Build the figure
		fig_title = "Sample variates"
		p = figure(width=400, height=300, title=fig_title, **kwargs)
		p.toolbar_location = None
		p.scatter(sample_index, sample_vals, size=4.0, color="gray", legend_label="samples")
		p.legend.location = "top_left"
		p.yaxis.axis_label = f"Estimates ({(self.confidence*100):.0f}% CI band)"
		p.xaxis.axis_label = "Sample index"
		if mu is not None:
			true_sp = Span(location=mu, dimension="width", line_dash="solid", line_color="red", line_width=1.0)
			p.add_layout(true_sp)
		p.line(sample_index, sample_avgs, line_color="black", line_width=2.0, legend_label="mean estimate")

		## Add confidence band
		band_source = ColumnDataSource(
			dict(x=sample_index, lower=sample_avgs - cum_abs_error, upper=sample_avgs + cum_abs_error)
		)
		conf_band = Band(
			base="x",
			lower="lower",
			upper="upper",
			source=band_source,
			fill_alpha=0.3,
			fill_color="yellow",
			line_color="black",
		)
		p.add_layout(conf_band)

		## Error plot
		error_title = "Estimator accuracy"
		# if isinstance(samples, dict):
		#   error_title += f" (converged: {np.take(samples['convergence']['converged'], 0)})"
		q1 = figure(width=300, height=150, y_axis_location="left", title=error_title)
		q2 = figure(width=300, height=150, y_axis_location="left")
		q1.toolbar_location = None
		q2.toolbar_location = None
		q1.yaxis.axis_label = "Rel. std-dev (CV)"
		q2.yaxis.axis_label = f"Abs. error ({(self.confidence*100):.0f}% CI)"
		q2.xaxis.axis_label = "Sample index"
		q1.yaxis.formatter = NumeralTickFormatter(format="0.00%")
		q1.y_range = Range1d(0, np.ceil(max(cum_rel_error) * 100) / 100, bounds=(0, 1))
		q2.x_range = q1.x_range = Range1d(0, len(sample_index))
		# if isinstance(samples, dict):
		#   q1.add_layout(BoxAnnotation(top=100, bottom=0, left=0, right=min_samples, fill_alpha=0.4, fill_color='#d3d3d3'))
		#   q2.add_layout(BoxAnnotation(top=100, bottom=0, left=0, right=min_samples, fill_alpha=0.4, fill_color='#d3d3d3'))

		## Plot the relative error
		rel_error_line = q1.line(sample_index, cum_rel_error, line_width=1.5, line_color="gray")

		## Plot the absolute error + its range
		# q.extra_y_ranges = {"abs_error_rng": Range1d(start=0, end=np.ceil(max(cumulative_abs_error)))}
		# q.add_layout(LinearAxis(y_range_name="abs_error_rng"), 'right')
		# q.yaxis[1].axis_label = "absolute error"
		abs_error_line = q2.line(sample_index, cum_abs_error, line_color="black")

		## Show the thresholds for convergence towards the thresholds
		rel_error_threshold = q1.line(
			x=[0, sample_index[-1]], y=[self.rtol, self.rtol], line_dash="dotted", line_color="gray", line_width=1.0
		)
		abs_error_threshold = q2.line(
			x=[0, sample_index[-1]], y=[self.atol, self.atol], line_dash="dashed", line_color="darkgray", line_width=1.0
		)

		## Add the legend
		legend_items = [("atol", [abs_error_line]), ("rtol", [rel_error_line])]
		# legend_items += [('abs threshold', [abs_error_threshold])] + [('rel threshold', [rel_error_threshold])]
		legend = Legend(items=legend_items, location="top_right", orientation="horizontal", border_line_color="black")
		legend.label_standoff = 1
		legend.label_text_font_size = "10px"
		legend.padding = 2
		legend.spacing = 5

		# q1.add_layout(legend, "center")
		return row([p, column([q1, q2])])
