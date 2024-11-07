import numpy as np
from array import array
from typing import Union, Callable
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator
from numbers import Number, Real
from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal
from scipy.sparse.linalg import eigsh
import bisect

from .trace import hutch, xtrace
from .operator import matrix_function, ShiftedInvOp
from .diagonalize import lanczos, _lanczos
from .quadrature import sl_gauss
from .special import param_callable


## Since Python has no support for inline creation of sized generators
## Based on "Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a Random Start"
class RelativeErrorBound:
	def __init__(self, n: int):
		self.base_num = 2.575 * np.log(n)
		self._len = n

	def __len__(self):
		return self._len

	def __getitem__(self, i: int):
		return -self.base_num / i**2


def spectral_radius(A: Union[LinearOperator, np.ndarray], rtol: float = 0.01, full_output: bool = False):
	"""Estimates the spectral radius to within a relative tolerance."""
	assert len(A.shape) == 2 and len(np.unique(A.shape)) == 1, "A must be square"
	EPS = np.finfo(A.dtype).eps
	n = A.shape[0]
	deg_bound = n
	if n < 150:
		rel_error_bound = 2.575 * np.log(n) / np.arange(4, n) ** 2
		deg_bound = max(np.searchsorted(-rel_error_bound, -rtol) + 5, 4)
	else:
		## This does binary search like searchsorted but uses O(1) memory
		re_bnd = RelativeErrorBound(n)
		deg_bound = max(bisect.bisect_left(re_bnd, -rtol) + 1, 4)
	k: int = min(deg_bound, n - 1)
	v0 = np.random.uniform(size=n)
	v0 /= np.linalg.norm(v0)
	(a, b) = lanczos(A, v0=v0, deg=k - 1, return_basis=False)
	rw = _lanczos.ritz_values(a, np.append(0.0, b), k)
	if not full_output:
		return np.max(rw)
	else:
		sr = np.max(rw)  # spectral radius estimate
		tol = sr * n * EPS  # NumPy default tolerance
		# rw_diffs = np.abs(np.diff(rw)/np.where(rw[1:] <= 0, 1.0, rw[1:]))
		# np.argmax(np.diff(x) / np.where(x[1:] <= 0, 1.0, x[1:])) + 1
		# rw[np.argmax(rw_diffs)+1]
		if full_output:
			out = {
				"gap": np.min(rw),
				"tolerance": tol,
				"ritz_values": rw,
				"deg_bound": deg_bound,
				"rtol": rtol,
				"spectral_radius": sr,
			}
		return sr, out


def spectral_density(
	A: Union[LinearOperator, np.ndarray],
	fun: Union[str, Callable] = None,
	bins: Union[int, np.ndarray] = 100,
	bw: Union[float, str] = "scott",
	deg: int = 20,
	rtol: float = 0.01,
	verbose: bool = False,
	info: bool = False,
	plot: bool = False,
	**kwargs,
):
	"""Estimates the spectral density of an operator via stochastic Lanczos quadrature.

	Parameters:
		A = LinearOperator
		bins = number of domain points to accumulate density
		bw = bandwidth value or rule
		deg = degree of each quadrature approximation
		rtol = relative stopping tolerance
		verbose = whether to report various statistics

	Return:
		(density, bins) = Estimate of the spectral density at domain points 'bins'
	"""
	## First probe info about the spectrum via a single adaptive Krylov expansion
	n = A.shape[0]
	spec_radius, info = spectral_radius(A, full_output=True)
	min_rw = np.min(info["ritz_values"])
	fun = "identity" if fun is None or (isinstance(fun, str) and fun == "identity") else fun
	fun = param_callable(fun, kwargs) if isinstance(fun, str) else fun
	assert isinstance(fun(1.0), Number), "Function must return a real number."

	## Parameterize the kernel
	## Automatic bandwidth determination for "bimodal or multi-modal distributions tend to be oversmoothed."
	N = deg * n
	if bw == "scott":
		h = N ** (-1 / 5)
		h **= 2  # to prevent over-smoothing
	elif bw == "silverman":
		h = (N * 3 / 4) ** (-1 / 7)
		h **= 2  # to prevent over-smoothing
	else:
		assert isinstance(bw, Number), f"Invalid bandwidth estimator '{bw}'; must be 'scott', 'silverman', or float."
		h = bw
	K = lambda u: np.exp(-0.5 * u**2)

	## Prepare the bins for the estimate
	bins = np.linspace(min_rw, spec_radius, int(bins)) if isinstance(bins, Number) else np.asarray(bins)
	n_bins = len(bins)
	spectral_density = np.zeros(n_bins)  # accumulate density estimate
	density_residual = np.zeros(n_bins)  # difference in density per iteration
	min_bins = np.inf * np.ones(n_bins)  # min value encountered per bin

	## Begin sampling stochastic quadrature estimates
	rel_change, jj = np.inf, 0
	trace_samples = array("f")
	while rel_change > rtol and jj < A.shape[0]:
		## Acquire a quadrature estimate
		## TODO: The inner sum can likely be vectorized with an einsum or something
		nodes, weights = sl_gauss(A, n=1, deg=deg, **kwargs).T
		density_residual.fill(0)
		for i, t in enumerate(nodes):
			density_residual += weights[i] * K((bins - t) / h)  # weights[i] * c # Note constant 'c' can be dropped

		# density_residual = np.sum(weights * K((bins[:,np.newaxis] - nodes) / h), axis=1)
		# np.sum(weights * K((bins[:,np.newaxis] - nodes) / h), axis=1)
		# np.sum(weights * (bins[:,np.newaxis] - nodes), axis=1)
		# np.einsum('i,ji,j->j', weights, bins[:, np.newaxis] - nodes, np.ones_like(bins))

		## Maintain a minimum ritz estimate per bin to estimate spectral gap
		bin_ind = np.clip(np.digitize(nodes, bins), 0, n_bins - 1)
		min_bins[bin_ind] = np.minimum(min_bins[bin_ind], nodes)

		## Accumulate the spectral density
		spectral_density += density_residual
		jj += 1
		if jj > 2:
			w1 = (spectral_density - density_residual) / np.sum(spectral_density - density_residual)
			w2 = spectral_density / np.sum(spectral_density)
			rel_change = np.mean(np.abs((w1 - w2) / np.where(w1 > 0, w1, 1.0)))

		## Keep trace of the spectral sum each iteration
		trace_samples.append(np.sum(weights * fun(nodes) * n))

	## Normalize such it density approx. integrates to 1
	spectral_density /= np.sum(spectral_density) * np.diff(bins[:2])

	## Plot if requested
	if plot:
		from bokeh.plotting import figure, show

		p = figure(width=700, height=300, title=f"Estimated spectral density (bw = {h:.4f}, n_samples = {jj})")
		p.scatter(bins, spectral_density)
		p.line(bins, spectral_density)
		# y_lb = np.min(spectral_density) - np.ptp(spectral_density) * 0.025
		# p.scatter(ew, , marker='plus', color='red', fill_alpha=0.25, line_width=0, size=6)
		show(p)

	if info:
		info_dict = {
			"trace": np.mean(trace_samples),
			"rtol": rtol,
			"quad_est": trace_samples,
			"bandwidth": h,
			"n_samples": jj,
		}
		return (spectral_density, bins), info_dict
	return (spectral_density, bins)


def spectral_gap(A, gap_ub: float = None, atol: float = 1e-6, rtol: float = 0.5, shortcut: bool = False, **kwargs):
	# sr, info = spectral_radius(A, full_output=True)
	A, sr = normalize_spectrum(A)
	tol_est = np.max(A.shape) * np.finfo(A.dtype).eps
	gap_ub = 1.0 if gap_ub is None else (gap_ub / sr)
	# tol_est = info['tolerance']
	# tol_est = 1e-6 * sr

	# deg = kwargs.pop("deg", info['deg_bound'])
	# n = kwargs.pop("n", 30)
	# nodes, weights = sl_gauss(A, deg=deg, n=n, **kwargs).T
	# nodes = np.sort(nodes)
	# rdiffs = np.abs(np.diff(nodes)/nodes[:-1])

	# outliers = np.abs(rdiffs) > (np.mean(rdiffs) + 1.5*np.std(rdiffs))

	# ## Prefer first outlier
	# if np.any(outliers):
	# 	outlier_ind = np.flatnonzero(outliers)
	# 	gap_est = nodes[outlier_ind[0]]
	# else:
	# 	gap_est = nodes[np.argmax(rdiffs)+1]
	# best_min, gap_ub = np.inf, 0.0
	# while gap_ub < tol_est:
	# 	gap_ub = eigsh(op, which='LM', k=1, sigma=op.sigma, OPinv=op, return_eigenvectors=False).take(0)
	# 	op.sigma = 10 * op.sigma
	# 	best_min = min(best_min, gap_ub)

	## First get an upper bound on the gap
	op = ShiftedInvOp(A, sigma=tol_est)
	best_min = gap_ub
	## Now do binary search on the an interval [tol, gap_ub] until atol is satisfied
	## or until a min is found twice
	lb, ub = (tol_est, gap_ub)
	while np.abs(lb - ub) > atol:
		mid = lb + np.abs(ub - lb) / 2
		op.sigma = mid
		gap_est_bs = eigsh(op, which="LM", k=1, sigma=op.sigma, OPinv=op, return_eigenvectors=False, tol=rtol).take(0)
		print(f"[{lb}, {ub}] ({mid}) -> {gap_est_bs}, (w/ tol={tol_est})")
		if gap_est_bs < tol_est:
			diff = np.abs(lb - ub)
			lb = ub.copy()
			ub = ub + diff
		else:
			ub = mid
		if gap_est_bs > tol_est:
			if shortcut and np.isclose(gap_est_bs, best_min):
				break
			best_min = gap_est_bs
	return best_min * sr


def normalize_spectrum(A: Union[np.ndarray, sparray, LinearOperator], radius: float = None, **kwargs):
	"""Normalizes a given [matrix|array|operator] to have eigenvalues in the interval [-1, 1]"""
	is_ndarray = isinstance(A, np.ndarray)
	is_sparray = isinstance(A, sparray) or isinstance(A, spmatrix)
	is_linop = isinstance(A, LinearOperator)
	assert is_ndarray or is_sparray or is_linop, "Must provide a valid linear operator"
	sr = spectral_radius(A, **kwargs) if radius is None else float(radius)
	if np.isclose(sr, 1.0):
		return A, sr
	elif is_ndarray:
		B = A / sr
		return B, sr
	elif is_sparray:
		assert hasattr(A, "data"), "Sparse matrix must have .data member"
		B = A.copy()
		B.data /= sr
		return B, sr
	else:
		from scipy.sparse.linalg._interface import _ScaledLinearOperator

		# from scipy.sparse.linalg import LinearOperator, aslinearoperator
		B = _ScaledLinearOperator(A, alpha=1.0 / sr)
		return B, sr

	# from KDEpy import FFTKDE
	# nodes, weights = sl_gauss(A, n=100, deg=deg).T
	# kde = FFTKDE(kernel='epa', bw="ISJ").fit(nodes)
	# x, y = kde.evaluate()
	# p = figure(width=300, height=250)
	# p.line(x, y)
	# p.scatter(x,y)
	# show(p)
	# f = CubicSpline(x[x >= 0], y[x >= 0])
	# critical_points = f(f.derivative(1).roots())
	# f(critical_points[critical_points > tol])


def numrank(
	A: Union[LinearOperator, np.ndarray], est: str = "hutch", gap: Union[float, str] = "auto", psd: bool = True, **kwargs
):
	"""Estimates the numerical rank of a given operator via stochastic trace estimation.

	Parameters
	----------
	A : LinearOperator or sparray or ndarray
			The operator or matrix to estimate the numerical rank of.
	est : str
			The trace estimator to use.
	gap : str or float
			Lower bound on the magnitude of the smallest non-zero eigenvalue, or the 'spectral gap'.
	gap_rtol : float
			Relative
	psd : bool
			Whether `A` is a assumed positive semi-definite.

	Returns
	-------
	:
			Stochastic estimate of the numerical rank.
	"""
	assert hasattr(A, "shape"), "Must be properly shaped operator"
	if np.prod(A.shape) == 0:
		return 0
	## Use lanczos to get basic estimation of largest and smallest positive eigenvalues
	## Relative error bounds based on: the Largest Eigenvalue by the Power and Lanczos Algorithms with a Random Starts
	## Spectral gap bounds based on sec. 13.2 of "The Symmetric Eigenvalue Problem" by Paige and the by
	## comments by Lin in "APPROXIMATING SPECTRAL DENSITIES OF LARGE MATRICES"
	sr, info = spectral_radius(A, full_output=True)
	A, sr = normalize_spectrum(A, radius=sr)

	## Augment the degree based on the spectral radius bound
	kwargs["deg"] = kwargs.pop("deg", info["deg_bound"] + 20)

	## By default, estimate numerical rank to within rounding accuracy
	## Caps the number of the iterations using SciPy / ARPACK's heuristic
	kwargs["maxiter"] = kwargs.pop("maxiter", 10 * A.shape[0])  ## scipy's default for eigsh

	## In all situations, we should stop once we're within rounding range
	kwargs["atol"] = kwargs.pop("atol", 0.50)

	## Set the gap to the default tolerance in numpy
	gap = info["tolerance"] if gap == "auto" else float(gap)

	# default_kwargs = {}
	# if gap == "auto" or gap == "simple":
	# 	EPS = np.finfo(A.dtype).eps
	# 	deg = max(kwargs.get("deg", 20), 4)
	# 	n = A.shape[0]
	# 	if n < 150:
	# 		rel_error_bound = 2.575 * np.log(A.shape[0]) / np.arange(4, n)**2
	# 		deg_bound = max(np.searchsorted(-rel_error_bound, -gap_rtol) + 5, deg)
	# 	else:
	# 		## This does binary search like searchsorted but uses O(1) memory
	# 		re_bnd = RelativeErrorBound(n)
	# 		deg_bound = bisect.bisect_left(re_bnd, -gap_rtol) + 1 # , deg)

	# 	## Use PSD-specific theory to estimate spectral gap
	# 	a,b = lanczos(A, deg=deg_bound)
	# 	if psd:
	# 		if gap == "auto":
	# 			# from scipy.linalg import eigh_tridiagonal
	# 			rr, rv = eigh_tridiagonal(a,b)
	# 			tol = np.max(rr) * A.shape[0] * EPS # NumPy default tolerance
	# 			min_id = np.flatnonzero(rr >= tol)[np.argmin(rr[rr >= tol])] # [0,n]
	# 			coeff = b[min_id-1] if min_id == len(b) else min([b[min_id-1], b[min_id]])
	# 			gap = rr[min_id] - coeff * np.abs(rv[-1,min_id])
	# 			gap = rr[min_id] if gap < 0 else gap
	# 		elif gap == "simple":
	# 			## This is typically a better estimate of the gap, but has little theory
	# 			rr = eigvalsh_tridiagonal(a,b)
	# 			tol = np.max(rr) * A.shape[0] * EPS
	# 			denom = np.where(rr[:-1] == 0, 1.0, rr[:-1])
	# 			gap = max(rr[np.argmax(np.diff(rr) / denom) + 1], tol)
	# 	else:
	# 		rr = eigvalsh_tridiagonal(a,b)
	# 		gap = np.max(rr) * A.shape[0] * EPS # NumPy default tolerance
	# 		tol = A.shape[0] * EPS
	# 	default_kwargs.update(dict(fun="smoothstep", a=tol, b=gap))
	# else:
	# 	assert isinstance(gap, Number), "Threshold `eps` must be a number"
	# 	default_kwargs.update(dict(fun="numrank", threshold=gap))

	## Estimate numerical rank
	est = hutch(A, fun="numrank", threshold=gap, **kwargs)
	# if est == "hutch":
	# 	est = hutch(A, **kwargs)
	# elif est == "xtrace":
	# 	M = matrix_function(A, **kwargs)
	# 	est = xtrace(M)
	# else:
	# 	raise ValueError(f"Invalid estimator '{est}' provided.")
	return int(np.round(est)) if isinstance(est, Number) else est


# from scipy.optimize import minimize_scalar
# 		from scipy.interpolate import CubicSpline
# 		from more_itertools import run_length
# 		tol = spec_radius * n * np.finfo(A.dtype).eps

# 		## This can be useful for estimating the lower-bound needed to THRESHOLD where the gap is, but not
# 		## the actual gap itself! also the inflection point reflection technique doesn't work
# 		density_f = CubicSpline(bins, spectral_density)

# 		## Estimate the center of the spectral gap by estimating where first derivative is 0
# 		is_decreasing = np.diff(spectral_density) < 0
# 		gap_est1 = np.inf
# 		if np.any(is_decreasing):
# 			mono_segments = list(run_length.encode(is_decreasing))
# 			i,j = (mono_segments[0][1], mono_segments[1][1]) if len(mono_segments) >= 2 else (0, (n_bins-1) // 2)
# 			res = minimize_scalar(density_f, bracket=(bins[0], bins[i+j]))
# 			inflection_pt = 0.0
# 			if res.success:
# 				inflection_pt = res.x
# 			else:
# 				inflection_pt = np.min(np.maximum(density_f.derivative(1).roots(), bins[i]))
# 			possible_gaps = min_bins[min_bins != np.inf]
# 			gap_est1 = np.min(possible_gaps[inflection_pt < possible_gaps]) if np.any(inflection_pt < possible_gaps) else np.min(possible_gaps)
# 		gap_ests = min_bins[min_bins != np.inf]
# 		gap_est2 = np.min(gap_ests[gap_ests > tol])
# 		print(f"Gap est 2: {gap_est2}, tol: {tol}, ests: {gap_ests}")
# 		gap_est = min(gap_est1, gap_est2)
