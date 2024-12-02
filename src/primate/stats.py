from numbers import Number
import numpy as np
import scipy as sp


class Covariance:
	"""Updateable covariance matrix.

	Uses Welford's algorithm to stably update the sample mean and (co)variance estimates.
	"""

	def __init__(self, dim: int = 1):
		self.dim = dim
		self.n = 0
		self.mu = np.zeros(dim)
		self.S = np.zeros((dim, dim))

	@property
	def mean(self):
		if self.n == 0:
			return np.nan
		return self.mu.item() if self.dim == 1 else self.mu

	def update(self, X: np.ndarray) -> None:
		"""Update mean and (co)variance estimates based on new observations.

		Parameters:
			X: (batch_size, dim)-array representing new observations
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
		mean, std_err, m = np.mean(a), sp.stats.sem(a, ddof=1), sp.stats.t.ppf((1 + confidence) / 2.0, len(a) - 1)
		return mean - m * std_err, mean + m * std_err
	elif sdist == "normal":
		sq_n = np.sqrt(len(a))
		mean, std = np.mean(a), np.std(a, ddof=1)
		return sp.stats.norm.interval(confidence, loc=mean, scale=std / sq_n)
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


# class DeltaToleranceEstimator(ConvergenceEstimator):
# 	def __init__(self, rtol: float = 0.01, atol: float = 1.49e-08, ord: Union[int, str] = 2) -> None:
# 		super().__init__()
# 		self.rtol = rtol
# 		self.atol = atol
# 		self.ord = ord
# 		self.n_samples = 0
# 		self._val = None
# 		self._err = np.inf

# 	def converged(self) -> bool:
# 		if self._val is None:
# 			return False
# 		return self._err < self.atol or self._err < self.rtol * np.linalg.norm(self._val, ord=self.ord)

# 	def update(self, x: np.ndarray):
# 		if self._val is None:
# 			self._val = np.full(shape=x.shape, fill_value=np.inf)
# 		self._err = np.linalg.norm(x - self._val, ord=self.ord)
# 		self._val = x
# 		self.n_samples += 1

# 	@property
# 	def estimate(self) -> float:
# 		return self._val

# 	def __len__(self) -> int:
# 		return self.n_samples
