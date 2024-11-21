from dataclasses import dataclass, field
from numbers import Number
from typing import Callable, Optional, Protocol, Sized, Union, runtime_checkable

import numpy as np
import scipy as sp

from .stats import Covariance


@runtime_checkable
class ConvergenceEstimator(Sized, Protocol):
	"""Protocol for generic stopping criteria for sequences."""

	def __len__(self) -> int: ...
	def update(self, x: Union[float, np.ndarray], **kwargs: dict): ...
	def estimate(self) -> Union[float, np.ndarray]: ...
	def converged(self) -> bool: ...


@dataclass
class EstimatorResult:
	estimate: Union[float, np.ndarray]
	converged: bool = False
	status: str = ""
	nit: int = 0
	samples: list = field(default_factory=list)

	def update(self, est: ConvergenceEstimator, sample: Optional[float] = None):
		self.estimate = est.estimate
		self.converged = est.converged()
		self.status = est.__repr__()
		self.nit = est.__len__()
		if sample is not None:
			self.samples.append(sample)


class ToleranceEstimator(ConvergenceEstimator):
	def __init__(self, rtol: float = 0.01, atol: float = 1.49e-08, ord: Union[int, str] = 2) -> None:
		super().__init__()
		self.rtol = rtol
		self.atol = atol
		self.ord = ord
		self.n_samples = 0
		self.error = np.inf
		self.n_samples = 0
		self.mean = None

	def __len__(self) -> int:
		return self.n_samples

	def update(self, x: Union[float, np.ndarray]):
		x = np.atleast_1d(x)
		x = x[:, None] if x.ndim == 1 else x
		# assert X.shape[1] == self.dim, f"Expected shape (n, {self.dim}), got {X.shape}"

		if self.mean is None:
			self.mean = np.zeros(shape=x.shape[1])
		delta_mean = x.mean(axis=0) - self.mean
		new_n = self.n_samples + x.shape[0]
		new_mean = self.mean + (x.shape[0] / new_n) * delta_mean
		# self.mean += (X.shape[0] / new_n) * delta_mean
		self.error = np.linalg.norm(new_mean - self.mean, ord=self.ord)
		self.mean = new_mean
		self.n_samples = new_n

	@property
	def estimate(self) -> float:
		return self.mean.item() if len(self.mean) == 1 else self.mean.ravel()

	def converged(self) -> bool:
		if self.mean is None:
			return False
		return self.error < self.atol or self.error < self.rtol * np.linalg.norm(self.mean, ord=self.ord)


# class ConfidenceEstimator
# class UncertaintyEstimator
class ConfidenceEstimator(ConvergenceEstimator):
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
		self.z = 2 ** (1 / 2) * sp.special.erfinv(confidence)
		self.t_scores = sp.stats.t.ppf((confidence + 1.0) / 2.0, df=np.arange(30) + 1)
		self.margin_of_error = np.inf
		self.confidence = confidence
		self.cov = Covariance(dim=1)

	def __len__(self) -> int:
		return self.n_samples

	## Bulk update function, which keeps a running mean and
	def update(self, estimates: Union[float, np.ndarray, None] = None) -> "ConfidenceEstimator":
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

	@property
	def estimate(self):
		return self.cov.mean

	def converged(self) -> bool:
		if self.n_samples < 3:
			return False
		std_dev = self.cov.covariance() ** (1 / 2)
		std_error = std_dev / np.sqrt(self.n_samples)
		rel_error = abs(std_error / self.estimate)
		return self.margin_of_error <= self.atol or rel_error <= self.rtol

	def __repr__(self) -> str:
		msg = f"Est: {self.estimate:.3f} +/- {self.margin_of_error:.3f}"
		msg += f" ({self.confidence*100:.0f}% CI,"  # | {(cv*100):.0f}% CV
		msg += f" #S:{ self.n_samples })"
		return msg


class ControlVariableEstimator(ConfidenceEstimator):
	def __init__(self, ecv: Union[float, np.ndarray], alpha: Optional[Union[float, np.ndarray]] = None, **kwargs: dict):
		super().__init__(**kwargs)
		ecv = np.atleast_1d(ecv).ravel()
		assert alpha is None or len(ecv) == len(alpha), "Coefficients alpha must have same length as the control variables."
		self.ecv = ecv
		self.alpha = alpha
		self.cov = Covariance(dim=len(ecv) + 1)
		self._estimate_cor = alpha is None
		self.n_samples = 0

	def __len__(self) -> int:
		return self.n_samples

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

	@property
	def estimate(self):
		if self.n_samples == 0:
			return np.nan
		cv_est = self.cov.mean[0] - np.dot(self.alpha, self.cov.mean[1:] - self.ecv)
		# alpha = self.alpha if self.alpha is not None else (C[0, 1] / C[1, 1])
		# cv_est = self.cov.mean[0] - alpha * (self.cov.mean[1] - self.ev)
		return cv_est.item()

	def converged(self) -> bool:
		if self.n_samples < 3:
			return False
		score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
		SE = self.margin_of_error / score
		rel_error = abs(SE / self.estimate)
		return self.margin_of_error <= self.atol or rel_error <= self.rtol


class KneeEstimator:
	def __init__(self, increasing: bool = False, S: float = 1.0, transform: Optional[Callable] = None) -> None:
		self.S = S
		self.values = []
		self.increasing = increasing
		self.transform = transform

	def __len__(self):
		return len(self.values)

	def update(self, x: Union[float, np.ndarray]):
		y = np.atleast_1d(x)
		if self.transform is not None:
			y = self.transform(y)
		self.values.extend(y)
		# assert np.diff(y)

	def estimate(self) -> float:
		if not self.values:
			raise ValueError("No values tracked yet.")
		return self.values[-1]

	def converged(self):
		"""Applies the kneedle algorithm to detect the knee in the sequence."""
		if len(self.values) < 3:
			return False

		## Normalize to [0, 1], calculate difference curve
		y = np.array(self.values) if self.increasing else np.max(self.values) - np.array(self.values)
		y_norm = (y - y.min()) / (y.max() - y.min())
		diff_curve = y_norm - np.linspace(0, 1, len(y))

		## Find the maxima / knee
		max_diff_idx = np.argmax(diff_curve)
		max_diff = diff_curve[max_diff_idx]

		## Set the knee_detected flag if the knee is prominent enough
		threshold = max_diff - (self.S / (len(y) - 1))
		return max_diff > threshold and diff_curve[-1] < threshold
