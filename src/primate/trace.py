import os
from typing import Union, Callable, Any
from numbers import Integral
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve_triangular
from scipy.stats import t
from scipy.special import erfinv
from numbers import Real

## Package imports
from .random import _engine_prefixes, _engines, isotropic
from .special import _builtin_matrix_functions
from .operator import matrix_function, OrthComplement
import _lanczos
import _trace
import _orthogonalize

_default_tvals = t.ppf((0.95 + 1.0) / 2.0, df=np.arange(30)+1)

# from collections import namedtuple
# HutchParams = namedtuple('HutchParams', ['a', 'b'])

def _operator_checks(A: Any) -> None:
	attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
	assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method"
	assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
	assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"
	assert hasattr(A, "shape"), "Operator 'A' must have a valid 'shape' attribute!"

def _estimator_msg(info) -> str:
	msg = f"{info['estimator']} estimator"
	msg += f" (fun={info.get('function', None)}"
	if info.get('lanczos_kwargs', None) is not None:
		msg += f", deg={info['lanczos_kwargs'].get('deg', 20)}"
	if info.get('quad', None) is not None:
		msg += f", quad={info['quad']}"
	msg += ")\n"
	msg += f"Est: {info['estimate']:.3f}"
	if 'margin_of_error' in info:
		moe, conf, cv = (info[k] for k in ['margin_of_error', 'confidence', 'coeff_var'])
		msg += f" +/- {moe:.3f} ({conf*100:.0f}% CI | {(cv*100):.0f}% CV)" 
	msg += f", (#S:{ info['n_samples'] } | #MV:{ info['n_matvecs']}) [{info['pdf'][0].upper()}]"
	if info.get('seed', -1) != -1: 
		msg += f" (seed: {info['seed']})"
	return msg 

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
	quad: str = "golub_welsch",
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
	The estimator is obtained by averaging quadratic forms $v \\mapsto v^T f(A)v$, rescaling as necessary.

	:::{.callout-note}	
	Convergence behavior is controlled by the `stop` parameter: "confidence" uses the central limit theorem to generate confidence 
	intervals on the fly, which may be used in conjunction with `atol` and `rtol` to upper-bound the error of the approximation. 
	Alternatively, when `stop` = 'change', the estimator is considered converged when the error between the last two iterates is less than 
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
	quad: { 'golub_welsch', 'fttr' }, default='golub_welsch'
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
			Estimate the trace of $f(A)$. If `info = True`, additional information about the computation is also returned.

	Notes
	-----
	To compute the weights of the quadrature, `quad` can be set to either 'golub_welsch' or 'fttr'. The former uses implicit symmetric QR 
	steps with Wilkinson shifts, while the latter (FTTR) uses the explicit recurrence expression for orthogonal polynomials. While both require 
	$O(\\mathrm{deg}^2)$ time to execute, the former requires $O(\\mathrm{deg}^2)$ space but is highly accurate, while the latter uses 
	only $O(1)$ space at the cost of stability. If `deg` is large, `fttr` is preferred is performance, though pilot testing should be 
	done to ensure that instability does not cause a large bias in the approximation. 

	See Also
	--------
	lanczos : the lanczos tridiagonalization algorithm.

	Reference
	---------
	1. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
	2. Hutchinson, Michael F. "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." Communications in Statistics-Simulation and Computation 18.3 (1989): 1059-1076.
	"""
	## Quick + basic input validation checks
	_operator_checks(A)

	## Catch degenerate cases 
	if (np.prod(A.shape) == 0) or (np.sum(A.shape) == 0):
		return 0

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
	deg: int = N if deg < 0 else int(np.clip(deg, 1, N)) # 1 <= deg <= N
	ncv: int = int(np.clip(ncv, 2, min(deg, N)))				 # 2 <= ncv <= deg
	orth: int = int(max(0, np.clip(orth, 0, ncv - 2))) # orth <= (deg - 2)
	atol: float = 0.0 if atol is None else float(atol)  
	rtol: float = 0.0 if rtol is None else float(rtol) 
	## *should* be safe to pass <= 0 on C++ side, but for redundancy we use os.cpu_count()
	num_threads: int = os.cpu_count() if num_threads < 0 else int(num_threads) 
	assert ncv >= 2 and ncv >= (orth+2) and ncv <= deg, f"Invalid Lanczos parameters ncv={ncv}, orth={orth}, deg={deg}; (orth < ncv? {orth < ncv}, ncv >= 2 ? {ncv >= 2}, ncv <= deg? {ncv <= deg})"

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
		assert isinstance(fun(1.0), Real), "Spectral function must return a real-valued number"
		kwargs["matrix_func"] = fun
		num_threads = 1
	else:
		raise ValueError(f"Invalid matrix function type '{type(fun)}'")

	## Collect the arguments processed so far
	assert 0 < confidence and confidence < 1, "Confidence must be in (0, 1)"
	t_values = _default_tvals if confidence == 0.95 else t.ppf((confidence + 1.0) / 2.0, df=np.arange(30)+1)
	z = np.sqrt(2) * erfinv(confidence)
	hutch_args = (nv, distr_id, engine_id, seed, deg, 0.0, orth, ncv, quad_id, atol, rtol, num_threads, use_clt, t_values, z)

	## Make the actual call
	info_dict = _trace.hutch(A, *hutch_args, **kwargs)
	
	## Return as early as possible if no additional info requested for speed 
	if not verbose and not info and not plot:
		return info_dict["estimate"]

	## Post-process info dict 
	info_dict['estimator'] = "Girard-Hutchinson"
	info_dict['valid'] = info_dict['samples'] != 0
	info_dict['n_samples'] = np.sum(info_dict['valid'])
	info_dict['n_matvecs'] = info_dict['n_samples'] * deg
	info_dict['std_error'] = np.std(info_dict['samples'][info_dict['valid']], ddof=1) / np.sqrt(info_dict['n_samples'])
	info_dict['coeff_var'] = np.abs(info_dict['std_error'] / info_dict['estimate'])
	info_dict['margin_of_error'] = (t_values[info_dict['n_samples']] if info_dict['n_samples'] < 30 else z) * info_dict['std_error']
	info_dict['confidence'] = confidence
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

	## Print the status if requested
	if verbose: 
		print(_estimator_msg(info_dict))

	## Plot samples if requested
	if plot:
		from bokeh.plotting import show
		from .plotting import figure_trace
		p = figure_trace(info_dict["samples"])
		show(p)
		info_dict['figure'] = figure_trace(info_dict["samples"])

	## Final return 
	return (info_dict["estimate"], info_dict) if info else info_dict["estimate"]


def hutchpp(
	A: Union[LinearOperator, np.ndarray],
	fun: Union[str, Callable] = None, 
	nb: int = "auto",
	maxiter: Union[str, int] = "auto", 
	mode: str = 'reduced', 
	**kwargs
) -> Union[float, dict]:
	"""Hutch++ estimator. 
	
	"""
	## Catch degenerate cases 
	_operator_checks(A)
	if (np.prod(A.shape) == 0) or (np.sum(A.shape) == 0):
		return 0

	## Convert to a matrix function, if not already 
	A = matrix_function(A, fun=fun, **kwargs) if fun is not None else A

	## Setup constants 
	verbose, info = kwargs.get('verbose', False), kwargs.get('info', False)
	N: int = A.shape[0]
	nb = (N // 3) if nb == "auto" else nb													 # number of samples to dedicate to deflation
	maxiter: int = (N // 3) if maxiter == "auto" else int(maxiter) # residual samples; default rule uses Hutch++ result
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	info_dict = { }

	## Sketch Y / Q - use numpy for now, but consider parallelizing MGS later
	WB = np.random.choice([-1.0, +1.0], size=(N, nb)).astype(f_dtype)
	Q = np.linalg.qr(A @ WB, mode='reduced')[0]
	# Y = np.array(A @ W2, order='F')
	# assert Y.flags['F_CONTIGUOUS'] and Y.flags['OWNDATA'] and Y.flags['WRITEABLE']
	# _orthogonalize.mgs(Y, 0)
	# Q = Y # Q is mostly orthonormal

	## Estimate trace of the low-rank sketch
	tr_defl = 0.0
	if mode == 'full': 
		## Full mode may not be space efficient, but is potentially vectorized, so suitable for relatively small output dimen.
		## https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v
		defl_ests = np.einsum('...i,...i->...', A @ Q, Q)
		tr_defl = np.sum(defl_ests)
	else:
		## Uses at most O(n) memory, but potentially slower 
		tr_defl, defl_ests = _trace.quad_sum(A, Q) if fun is None else A.quad_sum(Q)

	## Estimate trace of the residual via Girard-Hutchinson on the complement of the deflated subspaces
	if fun is None or mode == 'full':
		G = np.random.choice([-1.0, +1.0], size=(N, maxiter)).astype(f_dtype)
		G -= Q @ (Q.T @ G)
		defl_ests = np.einsum('...i,...i->...', A @ G, G)
		tr_resi = (1 / maxiter) * np.sum(defl_ests)
		#tr_resi = (1 / m) * (G.T @ (A @ G)).trace()
	else: 
		A.deflate(Q)
		kwargs['info'] = True
		kwargs['verbose'] = False
		tr_resi, ID = hutch(A, maxiter=maxiter, **kwargs)
		info_dict.update(ID)
		
	## Modify the info dict
	deg = 1 if fun is None else A.deg
	print(f"Deflation: {tr_defl}, Residual: {tr_resi}")
	info_dict['estimate'] = tr_defl + tr_resi
	info_dict['estimator'] = 'Hutch++'
	info_dict['n_matvecs'] = 2*nb*deg + info_dict.get('n_samples', maxiter)*deg
	info_dict['n_samples'] = nb + info_dict.get('n_samples', maxiter)
	info_dict['pdf'] = 'rademacher'	

	## Print as needed 
	if verbose:
		print(_estimator_msg(info_dict))
	return info_dict['estimate'] if not info else (info_dict['estimate'], info_dict)

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
	fun: Union[str, Callable] = None,
	nv: Union[str, int] = "auto", 
	pdf: str = "sphere",
	atol: float = 0.0, 
	rtol: float = 0.0, 
	cond_tol: float = 1e8,
	verbose: int = 0,
	info: bool = False, 
	**kwargs
):
	assert atol >= 0.0 and rtol >= 0.0, "Error tolerances must be positive"
	assert cond_tol >= 0.0, "Condition number must be non-negative"
	nv = int(nv) if isinstance(nv, Integral) else int(np.ceil(np.sqrt(A.shape[0])))
	n = A.shape[0]

	## If fun is specified, transparently convert A to matrix function 
	if isinstance(fun, str):
		assert fun in _builtin_matrix_functions, "If given as a string, matrix_function be one of the builtin functions."
		A = matrix_function(A, fun=fun)
	elif isinstance(fun, Callable):
		A = matrix_function(A, fun=fun)
	elif fun is not None:
		raise ValueError(f"Invalid matrix function type '{type(fun)}'")

	## Setup outputs. TODO: these should really be resizable arrays
	Y, Om, Z = np.zeros(shape=(n, 0)), np.zeros(shape=(n, 0)), np.zeros(shape=(n, 0))
	t, err = np.inf if (rtol != 0) else 0, np.inf
	it = 0
	cond_numb_bound = np.inf
	while Y.shape[1] < A.shape[1]:  # err >= (error_atol + error_rtol * abs(t)):
		## Determine number of new sample vectors to generate
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
	
		if err <= atol:
			break 

	if info: 
		info = {"estimate": t, "samples": t_samples, "error": err }
		return t, info 
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