from dataclasses import dataclass, field
from operator import and_, or_
from typing import Callable, Optional, Protocol, Sized, Union, runtime_checkable

import numpy as np
import scipy as sp

from .stats import Covariance


@runtime_checkable
class Estimator(Sized, Protocol):
	"""Protocol for generic stopping criteria for sequences."""

	n_samples: int

	def __len__(self) -> int:
		return self.n_samples

	def update(self, x: Union[float, np.ndarray], **kwargs: dict): ...
	def estimate(self) -> Union[float, np.ndarray]: ...


class ConvergenceCriterion(Callable):
	"""Generic stopping criteria for sequences."""

	def __init__(self, operation: Callable):
		assert callable(operation)
		self._operation = operation

	def __or__(self, other: "ConvergenceCriterion"):
		# other_op = other._operation if isinstance(other, ConvergenceCriterion) else (lambda: other)
		return ConvergenceCriterion(lambda est: or_(self(est), other(est)))

	def __and__(self, other: "ConvergenceCriterion"):
		# other_op = other._operation if isinstance(other, ConvergenceCriterion) else (lambda: other)
		return ConvergenceCriterion(lambda est: and_(self(est), other(est)))

	def __call__(self, est: Estimator) -> bool:
		return self._operation(est)


@dataclass
class EstimatorResult:
	estimate: Union[float, np.ndarray]
	estimator: Estimator
	criterion: Union[ConvergenceCriterion, str, None] = None
	status: str = ""
	nit: int = 0
	info: dict = field(default_factory=dict)

	def update(self, est: Estimator, **kwargs: dict):
		self.estimate = est.estimate
		self.estimator = est
		self.status = est.__repr__()
		self.nit = est.__len__()
		self.info = self.info | kwargs


class MeanEstimator(Estimator):
	"""Sample mean estimator with stable covariance updating."""

	delta: Union[float, np.ndarray]
	cov: Optional[Covariance]
	values: Optional[list]

	def __init__(self, record: bool = False) -> None:
		self.n_samples = 0
		self.delta = None
		self.cov = None
		self.values = [] if record else None

	@property
	def mean(self) -> Optional[float]:
		if self.cov is None:
			return None
		return self.cov.mu.item() if len(self.cov.mu) == 1 else self.cov.mu.ravel()

	def update(self, x: Union[float, np.ndarray]):
		x = np.atleast_1d(x)
		x = x[:, None] if x.ndim == 1 else x
		if self.cov is None:
			self.cov = Covariance(x.shape[1])
			self.delta = np.full(x.shape[1], np.inf)
		old_mu = self.cov.mu.copy()
		self.cov.update(x)
		self.delta = self.cov.mu - old_mu
		self.n_samples += x.shape[0]
		if self.values is not None:
			self.values.extend(x)

	@property
	def estimate(self) -> float:
		return self.mean


class ControlVariableEstimator(MeanEstimator):
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
		# SE = np.sqrt((1.0 / self.n_samples) * C_inner)
		# score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
		# self.margin_of_error = (SE * score).item()
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

	# def converged(self) -> bool:
	# 	if self.n_samples < 3:
	# 		return False
	# 	score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
	# 	SE = self.margin_of_error / score
	# 	rel_error = abs(SE / self.estimate)
	# 	return self.margin_of_error <= self.atol or rel_error <= self.rtol


class CountCriterion(ConvergenceCriterion):
	"""Convergence criterion that returns TRUE when above a given count."""

	def __init__(self, count: int):
		self.count = count

	def __call__(self, est: MeanEstimator) -> bool:
		return len(est) >= self.count


class ToleranceCriterion(ConvergenceCriterion):
	def __init__(self, rtol: float = 0.01, atol: float = 1.49e-08, ord: Union[int, str] = 2) -> None:
		self.rtol = rtol
		self.atol = atol
		self.ord = ord

	def __call__(self, est: MeanEstimator) -> bool:
		if est.mean is None:
			return False
		error = np.linalg.norm(est.delta, ord=self.ord)
		return error < self.atol or error < self.rtol * np.linalg.norm(est.mean, ord=self.ord)


class ConfidenceCriterion(ConvergenceCriterion):
	"""Parameterizes an expected value estimator that checks convergence of a sample mean within a confidence interval using the CLT.

	Provides the following methods:
		- __call__ = Updates the estimator with newly measured samples
		- converged = Checks convergence of the estimator within an interval
		-	plot = Plots the samples and their sample distribution CI's

	"""

	def __init__(self, confidence: float = 0.95, atol: float = 0.05, rtol: float = 0.01) -> None:
		assert 0 < confidence and confidence < 1, "Confidence must be in (0, 1)"
		self.atol = 0.0 if atol is None else atol
		self.rtol = 0.0 if rtol is None else rtol
		self.z = 2 ** (1 / 2) * sp.special.erfinv(confidence)
		self.t_scores = sp.stats.t.ppf((confidence + 1.0) / 2.0, df=np.arange(30) + 1)
		self.confidence = confidence

	## Bulk update function, which keeps a running mean and
	# def update(self, estimates: Union[float, np.ndarray, None] = None) -> "ConfidenceEstimator":
	# 	if estimates is None:
	# 		return self.converged()
	# 	self.cov.update(estimates)
	# 	self.n_samples = self.cov.n
	# 	var = self.cov.covariance()
	# 	std_dev = var ** (1 / 2)
	# 	score = self.t_scores[self.n_samples] if self.n_samples < 30 else self.z
	# 	SE = std_dev / float(self.n_samples) ** (1 / 2)
	# 	self.margin_of_error = score * SE  # todo: remove sqrt's
	# 	return self
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
	def __call__(self, est: MeanEstimator) -> bool:
		if est.n_samples < 3:
			return False
		std_dev = est.cov.covariance() ** (1 / 2)
		std_error = std_dev / np.sqrt(est.n_samples)
		rel_error = abs(std_error / est.estimate)
		score = self.t_scores[est.n_samples] if est.n_samples < 30 else self.z
		margin_of_error = score * std_error
		return margin_of_error <= self.atol or rel_error <= self.rtol

	# def __repr__(self) -> str:
	# 	msg = f"Est: {self.estimate:.3f} +/- {self.margin_of_error:.3f}"
	# 	msg += f" ({self.confidence*100:.0f}% CI,"  # | {(cv*100):.0f}% CV
	# 	msg += f" #S:{ self.n_samples })"
	# 	return msg


class KneeCriterion:
	def __init__(self, S: float = 1.0) -> None:
		self.S = S

	def __call__(self, est: MeanEstimator):
		"""Applies the kneedle algorithm to detect the knee in the sequence."""
		if est.values is None or len(est.values) < 3:
			return False

		## Extract the differences between the sample means
		mv = np.array(est.values).ravel()
		cum_sample_mean = mv / np.arange(1, len(mv) + 1)
		mu_cum_diffs = np.cumsum(np.abs(np.diff(cum_sample_mean)))

		## Normalize to [0, 1], calculate difference curve
		y = mu_cum_diffs
		y_norm = (y - y.min()) / (y.max() - y.min())
		diff_curve = y_norm - np.linspace(0, 1, len(y))

		## Find the maxima / knee
		max_diff_idx = np.argmax(diff_curve)
		max_diff = diff_curve[max_diff_idx]

		## Set the knee_detected flag if the knee is prominent enough
		threshold = max_diff - (self.S / (len(y) - 1))
		return max_diff > threshold and diff_curve[-1] < threshold


CRITERIA = {
	"count": CountCriterion,
	"tolerance": ToleranceCriterion,
	"confidence": ConfidenceCriterion,
	"knee": KneeCriterion,
}


def convergence_criterion(criterion: Union[str, ConvergenceCriterion], **kwargs) -> ConvergenceCriterion:
	# assert criterion.lower() in CRITERIA.keys(), f"Invalid criterion {criterion}"
	if isinstance(criterion, ConvergenceCriterion):
		return criterion
	criterion = criterion.lower()
	if criterion == "count":
		cc = CountCriterion(**{k: v for k, v in kwargs.items() if k in {"count"}})
	elif criterion == "tolerance":
		cc = ToleranceCriterion(**{k: v for k, v in kwargs.items() if k in {"ord", "atol", "rtol"}})
	elif criterion == "confidence":
		cc = ConfidenceCriterion(**{k: v for k, v in kwargs.items() if k in {"confidence", "atol", "rtol"}})
	elif criterion == "knee":
		cc = KneeCriterion(**{k: v for k, v in kwargs.items() if k in {"S"}})
	else:
		raise ValueError("Invalid criterion given")  # this should never happen
	assert isinstance(cc, ConvergenceCriterion), "`converge` must satisfy the ConvergenceEstimator protocol."
	return cc
