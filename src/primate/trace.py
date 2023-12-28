from typing import Union, Callable
from numbers import Integral
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve_triangular
from numbers import Real

## Package imports
from .random import _engine_prefixes, _engines, isotropic
from .special import _builtin_matrix_functions
import _lanczos
import _trace

def hutch(
	A: Union[LinearOperator, np.ndarray],
	fun: Union[str, Callable] = None,
	maxiter: int = 200,
	deg: int = 20,
	atol: float = None,
	rtol: float = None,
	stop: str = ["confidence", "change"],
	ncv: int = 2,
	orth: int = 0,
	quad: str = "fttr",
	confidence: float = 0.95,
	pdf: str = "rademacher",
	rng: str = "pcg64",
	seed: int = -1,
	num_threads: int = 0,
	verbose: bool = False,
	info: bool = False,
	plot: bool = False,
	**kwargs
) -> Union[float, tuple]:
	"""Estimates the trace of a symmetric $A$ or matrix function $f(A)$ via a Girard-Hutchinson estimator.

	This function uses up to `maxiter` random isotropic vectors to estimate of the trace of $f(A)$, where:
	$$\\mathrm{tr}(f(A)) = \\mathrm{tr}(U f(\\Lambda) U^T) = \\sum\\limits_{i=1}^n f(\\lambda_i) $$
	The estimator is obtained by averaging quadratic forms $v \mapsto v^T f(A)v$, rescaling as necessary.
	This estimator may be used to quickly approximate of a variety of quantities, such as the trace inverse, the log-determinant, the numerical rank, etc. 
	See the [online documentation](https://peekxc.github.io/primate/) for more details.

	:::{.callout-note}	
	Convergence behavior is controlled by the `stop` parameter: "confidence" uses the central limit theorem to generate confidence 
	intervals on the fly, which may be used in conjunction with `atol` and `rtol` to upper-bound the error of the approximation. 
	Alternatively, when `stop` = "change", the estimator is considered converged when the error between the last two iterates is less than 
	`atol` (or `rtol`, respectively), similar to the behavior of scipy.integrate.quadrature.
	:::

	Parameters
	----------
	A : ndarray, sparray, or LinearOperator
	    real symmetric operator.
	fun : str or Callable, default="identity"
	    real-valued function defined on the spectrum of `A`.
	maxiter : int, default=10
	    Maximum number of random vectors to sample for the trace estimate.
	deg  : int, default=20
	    Degree of the quadrature approximation. Must be at least 1. 
	atol : float, default=None
	    Absolute tolerance to signal convergence for early-stopping. See notes.
	rtol : float, default=1e-2
	    Relative tolerance to signal convergence for early-stopping. See notes.
	stop : str, default="confidence"
	    Early-stopping criteria to test estimator convergence. See details.
	ncv : int, default=2
			Number of Lanczos vectors to allocate. Must be at least 2. 
	orth: int, default=0
	    Number of additional Lanczos vectors to orthogonalize against. Must be less than `ncv`. 
	quad: { 'golub_welsch', 'fttr' }, default='fttr'
			Method used to obtain the weights of the Gaussian quadrature. See notes. 
	confidence : float, default=0.95
	    Confidence level to consider estimator as converged. Only used when `stop` = "confidence".
	pdf : { 'rademacher', 'normal' }, default="rademacher"
	    Choice of zero-centered distribution to sample random vectors from.
	rng : { 'splitmix64', 'xoshiro256**', 'pcg64', 'mt64' }, default="pcg64"
	    Random number generator to use.
	seed : int, default=-1
	    Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
	num_threads: int, default=0
	    Number of threads to use to parallelize the computation. Set to <= 0 to let OpenMP decide.
	plot : bool, default=False
	    If true, plots the samples of the trace estimate along with their convergence characteristics.
	info: bool, default=False
	    If True, returns a dictionary containing all relevant information about the computation.
	kwargs : dict, optional
	    additional key-values to parameterize the chosen function 'fun'.

	Returns
	-------
	:
			Estimate the trace of $f(A)$. If 'info = True', additional information about the computation is also returned.

	See Also
	--------
	lanczos : the lanczos tridiagonalization algorithm.

	Reference
	---------
	1. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
	"""
	attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
	assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method"
	assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
	assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"
	
	# from collections import namedtuple
	# HutchParams = namedtuple('HutchParams', ['a', 'b'])

	## Choose the random number engine
	assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
	engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)

	## Choose the distribution to sample random vectors from
	assert pdf in ["rademacher", "normal"], f"Invalid distribution '{pdf}'; Must be one of 'rademacher' or 'normal'."
	distr_id = ["rademacher", "normal"].index(pdf)

	## Choose the stopping criteria
	if stop == ["confidence", "change"] or stop == "confidence":
		use_clt: bool = True
		stop = "confidence"
	elif stop == "change":
		use_clt: bool = False
	else:
		raise ValueError(f"Invalid convergence criteria '{str(stop)}' supplied.")

	## Get the dtype; infer it if it's not available
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert (
		f_dtype.type == np.float32 or f_dtype.type == np.float64
	), "Only 32- or 64-bit floating point numbers are supported."

	assert quad in ["golub_welsch", "fttr"]
	quad_id = 0 if quad == "golub_welsch" else 1

	## Argument checking + clipping
	N: int = A.shape[0]
	nv: int = int(maxiter)  
	seed: int = int(seed)
	deg: int = int(np.clip(deg, 1, N))
	ncv: int = int(np.clip(ncv, 2, min(deg, N)))
	orth: int = int(min(deg if orth < 0 or orth > deg else orth, ncv - 1))
	atol: float = 0.0 if atol is None else float(atol)  
	rtol: float = 0.0 if rtol is None else float(rtol) 
	num_threads: int = 0 if num_threads < 0 else int(num_threads)
	assert ncv >= 2 and orth < ncv and ncv <= deg, f"Invalid Lanczos parameters (orth < ncv? {orth < ncv}, ncv >= 2 ? {ncv >= 2}, ncv <= deg? {ncv <= deg})"

	## Adjust tolerance for the quadrature estimates
	atol /= A.shape[1]
	assert not np.isnan(atol), "Absolute tolerance is NAN!"

	## Parameterize the matrix function and trace call
	if fun is None: 
		kwargs["function"] = "None"
	elif isinstance(fun, str):
		assert fun in _builtin_matrix_functions, "If given as a string, matrix_function be one of the builtin functions."
		kwargs["function"] = fun  # _builtin_matrix_functions.index(matrix_function)
	elif isinstance(fun, Callable):
		kwargs["function"] = "generic"
		assert isinstance(fun(0.0), Real), "Spectral function must return a real-valued number"
		kwargs["matrix_func"] = fun
	else:
		raise ValueError(f"Invalid matrix function type '{type(fun)}'")

	## Collect the arguments processed so far
	hutch_args = (nv, distr_id, engine_id, seed, deg, 0.0, orth, ncv, quad_id, atol, rtol, num_threads, use_clt)

	## Make the actual call
	info_dict = _trace.hutch(A, *hutch_args, **kwargs)
	
	## Print the status if requested
	if verbose: 
		from scipy.special import erfinv
		msg = f"Girard-Hutchinson estimator (fun={kwargs['function']}, deg={deg}, quad={quad})\n"
		valid_samples = info_dict['samples'] != 0
		n_valid = sum(valid_samples)
		std_error = np.std(info_dict['samples'][valid_samples], ddof=1) / np.sqrt(n_valid)
		z = np.sqrt(2.0) * erfinv(confidence)
		cv = np.abs(std_error / info_dict['estimate'])
		msg += f"Est: {info_dict['estimate']:.3f} +/- {z * std_error:.2f} ({confidence*100:.0f}% CI), CV: {(cv*100):.0f}%, " 
		msg += f"Evals: { n_valid } [{pdf[0].upper()}]"
		if seed != -1: 
			msg += f" (seed: {seed})"
		print(msg)

	## If only the point-estimate is required, return it
	if not info and not plot: 
		return info_dict["estimate"]
	
	## Otherwise build the info
	if plot:
		from bokeh.plotting import show
		from .plotting import figure_trace
		p = figure_trace(info_dict["samples"])
		show(p)
		info_dict['figure'] = figure_trace(info_dict["samples"])
	
	## Build the info dictionary 
	info_dict["stop"] = stop
	info_dict["pdf"] = pdf
	info_dict["rng"] = _engines[engine_id]
	info_dict["seed"] = seed
	info_dict["function"] = kwargs["function"]
	info_dict["lanczos_kwargs"] = dict(orth=orth, ncv=ncv, deg=deg)
	info_dict["quad"] = quad
	info_dict["rtol"] = rtol
	info_dict["atol"] = atol
	info_dict["num_threads"] = "auto" if num_threads == 0 else num_threads
	info_dict["maxiter"] = nv
	info_dict["confidence"] = confidence
	return info_dict["estimate"], info_dict


# TODO: implemented hutch++
# def hutchpp():

def sl_gauss(
	A: Union[LinearOperator, np.ndarray],
	n: int = 150,
	deg: int = 20,
	pdf: str = "rademacher",
	rng: str = "pcg",
	seed: int = -1,
	orth: int = 0,
	num_threads: int = 0,
) -> np.ndarray:
	"""Stochastic Gaussian quadrature approximation.

	Computes a set of sample nodes and weights for the degree-k orthogonal polynomial approximating the 
	cumulative spectral measure of `A`. This function can be used to approximate the spectral density of `A`, 
	or to approximate the spectral sum of any function applied to the spectrum of `A`.

	Parameters
	----------
	A : ndarray, sparray, or LinearOperator
	    real symmetric operator.
	n : int, default=150
	    Number of random vectors to sample for the quadrature estimate.
	deg  : int, default=20
	    Degree of the quadrature approximation.
	rng : { 'splitmix64', 'xoshiro256**', 'pcg64', 'lcg64', 'mt64' }, default="pcg64"
	    Random number generator to use (PCG64 by default).
	seed : int, default=-1
	    Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
	pdf : { 'rademacher', 'normal' }, default="rademacher"
	    Choice of zero-centered distribution to sample random vectors from.
	orth: int, default=0
		Number of additional Lanczos vectors to orthogonalize against when building the Krylov basis.
	num_threads: int, default=0
	    Number of threads to use to parallelize the computation. Setting `num_threads` < 1 to let OpenMP decide.

	Returns
	-------
	trace_estimate : float
	    Estimate of the trace of the matrix function $f(A)$.
	info : dict, optional
	    If 'info = True', additional information about the computation.

	"""
	attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
	assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method"
	assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
	assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"

	## Choose the random number engine
	assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
	engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)

	## Choose the distribution to sample random vectors from
	assert pdf in ["rademacher", "normal"], f"Invalid distribution '{pdf}'; Must be one of 'rademacher' or 'normal'."
	distr_id = ["rademacher", "normal"].index(pdf)

	## Get the dtype; infer it if it's not available
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert (
		f_dtype.type == np.float32 or f_dtype.type == np.float64
	), "Only 32- or 64-bit floating point numbers are supported."

	## Extract the machine precision for the given floating point type
	lanczos_rtol = np.finfo(f_dtype).eps  # if lanczos_rtol is None else f_dtype.type(lanczos_rtol)

	## Argument checking
	m = A.shape[1]  # Dimension of the space
	nv = int(n)  # Number of random vectors to generate
	seed = int(seed)  # Seed should be an integer
	deg = max(deg, 2)  # Must be at least 2
	orth = m - 1 if orth < 0 else min(m - 1, orth)  # Number of additional vectors should be an integer
	ncv = max(int(deg + orth), m)  # Number of Lanczos vectors to keep in memory
	num_threads = int(num_threads)  # should be integer; if <= 0 will trigger max threads on C++ side

	## Collect the arguments processed so far
	sl_quad_args = (nv, distr_id, engine_id, seed, deg, lanczos_rtol, orth, ncv, num_threads)

	## Make the actual call
	quad_nw = _lanczos.stochastic_quadrature(A, *sl_quad_args)
	return quad_nw


def __xtrace(W: np.ndarray, Z: np.ndarray, Q: np.ndarray, R: np.ndarray, method: str):
	"""Helper for xtrace function"""
	diag_prod = lambda A, B: np.diag(A.T @ B)[:, np.newaxis]

	## Invert R
	n, m = W.shape
	W_proj = Q.T @ W
	R_inv = solve_triangular(R, np.identity(m)).T # todo: replace with dtrtri?
	S = (R_inv / np.linalg.norm(R_inv, axis=0)) 

	## Handle the scale
	if not method == 'sphere':
	  scale = np.ones(m)[:,np.newaxis] # this is a column vector
	else:
	  col_norm = lambda X: np.linalg.norm(X, axis=0)
	  c = n - m + 1
	  scale = c / (n - (col_norm(W_proj)[:,np.newaxis])**2 + (diag_prod(S,W_proj) * col_norm(S)[:,np.newaxis])**2)

	## Intermediate quantities
	H = Q.T @ Z
	HW = H @ W_proj
	T = Z.T @ W
	dSW, dSHS = diag_prod(S, W_proj), diag_prod(S, H @ S)
	dTW, dWHW = diag_prod(T, W_proj), diag_prod(W_proj, HW)
	dSRmHW, dTmHRS = diag_prod(S, R - HW), diag_prod(T - H.T @ W_proj, S)

	## Trace estimate
	tr_ests = np.trace(H) * np.ones(shape=(m, 1)) - dSHS
	tr_ests += (-dTW + dWHW + dSW * dSRmHW + abs(dSW) ** 2 * dSHS + dTmHRS * dSW) * scale
	t = tr_ests.mean()
	err = np.std(tr_ests) / np.sqrt(m)
	return t, tr_ests, err

def xtrace(
	A: Union[LinearOperator, np.ndarray], 
	nv: Union[str, int] = "auto", 
	pdf: str = "sphere",
	atol: float = 0.1, 
	rtol: float = 1e-6, 
	cond_tol: float = 1e8,
	verbose: int = 0,
	info: bool = False
):
	assert atol >= 0.0 and rtol >= 0.0, "Error tolerances must be positive"
	assert cond_tol >= 0.0, "Condition number must be non-negative"
	nv = int(nv) if isinstance(nv, Integral) else int(np.ceil(np.sqrt(A.shape[0])))
	n = A.shape[0]
	Y, Om, Z = np.zeros(shape=(n, 0)), np.zeros(shape=(n, 0)), np.zeros(shape=(n, 0))
	t, err = np.inf if (rtol != 0) else 0, np.inf
	it = 0
	cond_numb_bound = np.inf
	while Y.shape[1] < A.shape[1]:  # err >= (error_atol + error_rtol * abs(t)):
		## Determine number of new vectors to sample
		ns = max(nv,2) if it == 0 else nv 
		ns = min(A.shape[1] - Y.shape[1], ns)
		
		## Sample a batch of random isotropic vectors
		NewOm = isotropic(size=(n, ns), method=pdf)
		tmp_Y, tmp_Om = np.c_[Y, A @ NewOm], np.c_[Om, NewOm]
		Q, R = np.linalg.qr(tmp_Y, "reduced") 
		# np.reshape((A @ NewOm)[:5,:], 10, order='F')
		
		## Expand the subspace
		Y, Om = tmp_Y, tmp_Om 
		Z = np.c_[Z, A @ Q[:, -ns:]]
		t, t_samples, err = __xtrace(Om, Z, Q, R, pdf)
		
		## Increase the iteration count and print, if warranted
		it += 1
		if verbose > 0: 
			print(f"It: {it}, est: {t:.8f}, Y_size: {Y.shape}, error: {err:.8f}")
	
	if info: 
		info = {"estimate": t, "samples": t_samples, "error": err }
		return info 
	return t


## TODO: Revisit what matrices cause degenerate xtrace examples
## If the sampled vectors have a lot of linear dependence, they won't (numerically) span a large enough subspace
## to permit sufficient exploration of the eigen-space, so we optionally re-sample based on a loose upper bound
## Based on: https://math.stackexchange.com/questions/1191503/conditioning-of-triangular-matrices
## A much cheaper option would be e.g. check the hamming distance between adjacent vectors or something
## maybe something like locality sensitive hashing would be better 
# while cond_numb_bound > cond_tol:
# 	R_mass = np.abs(np.diag(R))
# 	_cn = 3 * np.max(R_mass) / np.min(R_mass)
# 	cond_numb_bound = 0.0 if np.isclose(_cn, cond_numb_bound) else _cn
# 	if verbose > 1: 
# 		print(f"Condition number upper bound on sampling: {cond_numb_bound}")
# 	tmp_Y, tmp_Om = np.c_[Y, A @ NewOm], np.c_[Om, NewOm]
# 	Q, R = np.linalg.qr(tmp_Y, "reduced") 