from numbers import Number
from typing import Callable, Sized, Optional, Protocol, Union, runtime_checkable

import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, sem, t


class Covariance:
	"""Updateable covariance matrix.

	Uses Welford's online algorithm to update the sample mean and covariance estimates in a numerically stable way.
	"""

	def __init__(self, dim: int = 1):
		self.dim = dim
		self.n = 0
		self.mu = np.zeros(dim)
		self.S = np.zeros((dim, dim))

	@property
	def mean(self):
		return self.mu.item() if self.dim == 1 else self.mu

	def update(self, X: np.ndarray) -> None:
		"""Update mean and (co)variance estimates based on new observations.

		Parameters:
			X: (batch_size, dim)-ndarray representing new observations
		"""
		X = np.atleast_1d(X)
		X = X[:, None] if X.ndim == 1 else X
		assert X.shape[1] == self.dim, f"Expected shape (n, {self.dim}), got {X.shape}"

		## Compute batch mean and update overall mean
		batch_mean = X.mean(axis=0)
		delta_mean = batch_mean - self.mu
		new_n = self.n + X.shape[0]
		self.mu += (X.shape[0] / new_n) * delta_mean

		## Update sum of outer products
		X_centered = X - batch_mean
		X_shift = delta_mean[:, None] @ delta_mean[None, :] if self.dim > 1 else (delta_mean * delta_mean)
		self.S += (X_centered.T @ X_centered) + (self.n * X.shape[0] / new_n) * X_shift
		self.n = new_n

	def covariance(self, ddof: int = 1) -> np.ndarray:
		"""Covariance matrix of the observations.

		Parameters:
		  ddof: Delta degrees of freedom (1 for sample covariance, 0 for population)

		Returns:
		  Current covariance matrix estimate of shape (dim, dim)
		"""
		# assert ddof < self.n, f"Need more than {ddof} samples for ddof={ddof}"
		if (self.n - ddof) <= 0:
			return np.inf if self.dim else np.diag(np.inf, self.dim)
		cov = self.S / (self.n - ddof)
		return cov.item() if self.dim == 1 else cov


@runtime_checkable
class ConvergenceEstimator(Sized, Protocol):
	"""Protocol for generic stopping criteria for sequences."""

	def estimate(self) -> float: ...
	def update(self, **kwargs: dict): ...
	def converged(self) -> bool: ...


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


# def control_variate_estimator(samples: np.ndarray, cvs: np.ndarray, mu: float, alpha: Optional[float] = None):
# 	assert len(samples) == len(cvs), "Number of control variables must match number of samples."
# 	n = len(samples)
# 	if alpha is None:
# 		C = np.cov(samples, cvs, ddof=1)  # sample covariance
# 		alpha = C[0, 1] / C[1, 1]
# 	denom = np.arange(n)
# 	denom[0] = 1
# 	cv_est = (samples - alpha * (cvs - mu)) / denom
# 	# SE = sem(samples)
# 	C_inner = (1 - C[0, 1] ** 2 / np.prod(np.diag(C))) * C[0, 0]
# 	SE = np.sqrt((1 / n) * C_inner)
# 	z = norm.ppf(1.0 - (alpha / 2))
# 	return cv_est, (cv_est[-1] - z * SE, cv_est[-1] - z * SE)


class MeanEstimator(ConvergenceEstimator):
	def __init__(self, dim: int = 1) -> None:
		super().__init__()
		self.dim = dim
		self.n_samples = 0
		self.mean = np.zeros(dim)

	def converged(self) -> bool:
		return False

	def update(self, X: Union[float, np.ndarray]):
		X = np.atleast_1d(X)
		X = X[:, None] if X.ndim == 1 else X
		assert X.shape[1] == self.dim, f"Expected shape (n, {self.dim}), got {X.shape}"
		delta_mean = X.mean(axis=0) - self.mean
		new_n = self.n_samples + X.shape[0]
		self.mean += (X.shape[0] / new_n) * delta_mean
		self.n_samples = new_n

	@property
	def estimate(self) -> float:
		return self.mean.item() if self.dim == 1 else self.mean

	def __len__(self) -> int:
		return self.n_samples


class CentralLimitEstimator(ConvergenceEstimator):
	"""Parameterizes an expected value estimator that checks convergence of a sample mean within a confidence interval using the CLT.

	Provides the following methods:
		- __call__ = Updates the estimator with newly measured samples
		- converged = Checks convergence of the estimator within an interval
		-	plot = Plots the samples and their sample distribution CI's

	"""

	def __init__(self, confidence: float = 0.95, atol: float = 0.05, rtol: float = 0.01) -> None:
		assert 0 < confidence and confidence < 1, "Confidence must be in (0, 1)"
		self.n_samples = 0
		self.atol = 0.0 if atol is None else atol
		self.rtol = 0.0 if rtol is None else rtol
		self.z = 2 ** (1 / 2) * erfinv(confidence)
		self.t_scores = t.ppf((confidence + 1.0) / 2.0, df=np.arange(30) + 1)
		self.margin_of_error = np.inf
		self.confidence = confidence
		self.cov = Covariance(dim=1)

	@property
	def estimate(self):
		return self.cov.mean

	## Bulk update function, which keeps a running mean and
	def update(self, estimates: Union[float, np.ndarray, None] = None) -> "CentralLimitEstimator":
		if estimates is None:
			return self.converged()
		self.cov.update(estimates)
		self.n_samples = self.cov.n
		var = self.cov.covariance()
		std_dev = var ** (1 / 2)
		score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
		SE = std_dev / float(self.n_samples) ** (1 / 2)
		self.margin_of_error = score * SE  # todo: remove sqrt's
		return self
		# estimates = np.array([estimates]).ravel()
		# for estimate in estimates:
		# 	self.n_samples += 1
		# 	denom = 1.0 / float(self.n_samples)
		# 	L = float(self.n_samples - 2) / float(self.n_samples - 1) if self.n_samples > 2 else 0.0
		# 	self.mu_est = denom * (estimate + (self.n_samples - 1) * self.mu_pre)
		# 	self.mu_pre = self.mu_est if self.n_samples == 1 else self.mu_pre
		# 	self.vr_est = L * self.vr_pre + denom * (estimate - self.mu_pre) ** 2  # update sample variance
		# 	self.mu_pre = self.mu_est
		# 	self.vr_pre = self.vr_est

	def converged(self) -> bool:
		if self.n_samples < 3:
			return False
		std_dev = self.cov.covariance() ** (1 / 2)
		std_error = std_dev / np.sqrt(self.n_samples)
		rel_error = abs(std_error / self.estimate)
		return self.margin_of_error <= self.atol or rel_error <= self.rtol

	def __len__(self) -> int:
		return self.n_samples

	def __repr__(self) -> str:
		msg = f"Est: {self.estimate:.3f} +/- {self.margin_of_error:.3f}"
		msg += f" ({self.confidence*100:.0f}% CI,"  # | {(cv*100):.0f}% CV
		msg += f" #S:{ self.n_samples })"
		return msg


class ControlVariableEstimator(CentralLimitEstimator):
	def __init__(self, ecv: Union[float, np.ndarray], alpha: Optional[Union[float, np.ndarray]] = None, **kwargs: dict):
		super().__init__(**kwargs)
		ecv = np.atleast_1d(ecv).ravel()
		assert alpha is None or len(ecv) == len(alpha), "Coefficients alpha must have same length as the control variables."
		self.ecv = ecv
		self.alpha = alpha
		self.cov = Covariance(dim=len(ecv) + 1)
		self._estimate_cor = alpha is None
		self.n_samples = 0

	@property
	def estimate(self):
		if self.n_samples == 0:
			return np.nan
		cv_est = self.cov.mean[0] - np.dot(self.alpha, self.cov.mean[1:] - self.ecv)
		# alpha = self.alpha if self.alpha is not None else (C[0, 1] / C[1, 1])
		# cv_est = self.cov.mean[0] - alpha * (self.cov.mean[1] - self.ev)
		return cv_est.item()

	def update(self, samples: np.ndarray, cvs: np.ndarray):
		self.cov.update(np.c_[samples, cvs])
		self.n_samples = self.cov.n
		C = self.cov.covariance(ddof=1)
		C_00, C_01, C_11 = C[0, 0], C[1:, 0], C[1:, 1:]
		C_inner = (C_00 - C_01**2 / C_11) if self.cov.dim == 2 else (C_00 - np.dot(C_01, np.linalg.solve(C_11, C_01)))
		if self._estimate_cor:
			C_01, C_11 = C[1:, 0], C[1:, 1:]
			self.alpha = (C[0, 1] / C[1, 1]) if self.cov.dim == 2 else np.linalg.solve(C_11, C_01)
		SE = np.sqrt((1.0 / self.n_samples) * C_inner)
		score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
		self.margin_of_error = (SE * score).item()
		# self.r_sq = np.dot(C_01, np.linalg.solve(C_11, C_01)).item() / C_00
		return self
		## build the estimator
		# cv_est = np.mean(samples - alpha * (cvs - self.ev))
		# SE = sem(samples)
		# C_inner = (1 - C[0, 1] ** 2 / np.prod(np.diag(C))) * C[0, 0]
		# SE = np.sqrt((1 / n) * C_inner)
		# z = norm.ppf(1.0 - (alpha / 2))
		# return cv_est, (cv_est - z * SE, cv_est - z * SE)

	def converged(self) -> bool:
		if self.n_samples < 3:
			return False
		score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
		SE = self.margin_of_error / score
		rel_error = abs(SE / self.estimate)
		return self.margin_of_error <= self.atol or rel_error <= self.rtol
