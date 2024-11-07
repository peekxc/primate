from numbers import Integral, Real
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import sparray

from .special import param_callable
from .lanczos import _lanczos, _validate_lanczos, lanczos
from .quadrature import lanczos_quadrature
from .stats import MeanEstimatorCLT
from .stochastic import isotropic


## Based on Theorem 4.3 and Lemma 4.4 of Ubaru
def suggest_nv_trace(p: float, eps: float, f: str = "identity", dist: str = "rademacher") -> int:
	"""Suggests a number of sample vectors to use to get an eps-accurate trace estimate with probability p."""
	assert p >= 0 and p < 1, "Probability of success 'p' must be  must be between [0, 1)"
	eta = 1.0 - p
	if f == "identity":
		return int(np.round((6 / eps**2) * np.log(2 / eta)))
	else:
		raise NotImplementedError("TODO")


def _operator_checks(A: Union[sparray, np.ndarray, LinearOperator]) -> np.dtype:
	attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
	assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method"
	assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
	assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"
	assert hasattr(A, "shape"), "Operator 'A' must have a valid 'shape' attribute!"
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert f_dtype.type in {np.float32, np.float64}, "Only 32- or 64-bit floats are supported."
	return f_dtype


def _estimator_msg(info: dict) -> str:
	msg = f"{info['estimator']} estimator"
	msg += f" (fun={info.get('function', None)}"
	if info.get("lanczos_kwargs", None) is not None:
		msg += f", deg={info['lanczos_kwargs'].get('deg', 20)}"
	if info.get("quad", None) is not None:
		msg += f", quad={info['quad']}"
	msg += ")\n"
	msg += f"Est: {info['estimate']:.3f}"
	if "margin_of_error" in info:
		moe, conf, cv = (info[k] for k in ["margin_of_error", "confidence", "coeff_var"])
		msg += f" +/- {moe:.3f} ({conf*100:.0f}% CI | {(cv*100):.0f}% CV)"
	msg += f", (#S:{ info['n_samples'] } | #MV:{ info['n_matvecs']}) [{info['pdf'][0].upper()}]"
	if info.get("seed", -1) != -1:
		msg += f" (seed: {info['seed']})"
	return msg


def _reduce(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
	if nodes.ndim == 1:
		return np.sum(nodes * weights, axis=-1)
	return np.sum(nodes * weights[:, np.newaxis], axis=-1)


def hutch(
	A: Union[LinearOperator, np.ndarray],
	fun: Union[str, Callable, None] = None,
	reduce: Union[Callable, None] = None,
	maxiter: int = 200,
	deg: int = 20,
	atol: Optional[float] = None,
	rtol: Optional[float] = None,
	stop: str = "confidence",
	ncv: int = 2,
	orth: int = 0,
	quad: str = "gw",
	confidence: float = 0.95,
	pdf: str = "rademacher",
	seed: Optional[int] = None,
	num_threads: int = 0,
	verbose: bool = False,
	info: bool = False,
	plot: bool = False,
	**kwargs: dict,
) -> Union[float, tuple]:
	r"""Estimates the trace of a symmetric `A` or matrix function `fun(A)` via the Girard-Hutchinson estimator.

	This function uses up to `maxiter` random isotropic vectors to estimate of the trace of $f(A)$, where:
	$$ \mathrm{tr}(f(A)) = \mathrm{tr}(U f(\Lambda) U^T) = \sum_{i=1}^n f(\lambda_i), \quad A = U \Lambda U^T $$
	The estimator is obtained by averaging quadratic forms $v \mapsto v^T f(A)v$, rescaling as necessary.

	:::{.callout-note}
	Convergence behavior is controlled by the `stop` parameter: "confidence" uses the central limit theorem to generate confidence
	intervals on the fly, which may be used in conjunction with `atol` and `rtol` to upper-bound the error of the approximation.
	Alternatively, when `stop` = 'change', the estimator is considered converged when the error between the last two iterates is less than
	`atol` (or `rtol`, respectively), similar to the behavior of scipy.integrate.quadrature.
	:::

	Parameters:
		A: real symmetric operator.
		fun: real- or vector-valued function defined on the spectrum of `A`.
		reduce: per-estimate aggregation function. Required when `fun` is vector-valued. Default to identity.
		maxiter: Maximum number of random vectors to sample for the trace estimate.
		deg: Degree of the quadrature approximation. Must be at least 1.
		atol: Absolute tolerance to signal convergence for early-stopping. See notes.
		rtol: Relative tolerance to signal convergence for early-stopping. See notes.
		stop: Early-stopping criteria to test estimator convergence. See details.
		ncv: Number of Lanczos vectors to allocate. Must be at least 2.
		orth: Number of additional Lanczos vectors to orthogonalize against. Must be less than `ncv`.
		quad: Method used to obtain the weights of the Gaussian quadrature. See notes.
		confidence: Confidence level to consider estimator as converged. Only used when `stop` = "confidence".
		pdf: Choice of zero-centered distribution to sample random vectors from.
		seed: Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
		num_threads: Number of threads to use to parallelize the computation. Set to <= 0 to let OpenMP decide.
		plot: If true, plots the samples of the trace estimate along with their convergence characteristics.
		info: If True, returns a dictionary containing all relevant information about the computation.
		kwargs: additional key-values to parameterize the chosen function 'fun'.

	Returns:
		Estimate the trace of $f(A)$. If `info = True`, additional information about the computation is also returned.

	See Also:
		- [lanczos](/reference/lanczos.lanczos.md): the lanczos tridiagonalization algorithm.
		- [MeanEstimatorCLT](/reference/MeanEstimatorCLT.md): Standard estimator of the mean from iid samples.

	Notes:
		To compute the weights of the quadrature, `quad` can be set to either 'golub_welsch' or 'fttr'. The former uses implicit symmetric QR
		steps with Wilkinson shifts, while the latter (FTTR) uses the explicit recurrence expression for orthogonal polynomials. While both require
		$O(\mathrm{deg}^2)$ time to execute, the former requires $O(\mathrm{deg}^2)$ space but is highly accurate, while the latter uses
		only $O(1)$ space at the cost of stability. If `deg` is large, `fttr` is preferred for performance, though pilot testing should be
		done to ensure that instability does not cause a large bias in the approximation.

	Reference:
		1. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
		2. Hutchinson, Michael F. "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." Communications in Statistics-Simulation and Computation 18.3 (1989): 1059-1076.

	Examples:
		```{python}
		from primate.estimators import hutch
		```
	"""
	assert pdf in {"rademacher", "normal"}, f"Invalid distribution '{pdf}'; Must be one of 'rademacher' or 'normal'."
	assert stop in {"confidence", "change", None}
	assert quad in {"gw", "fttr"}
	assert 0 < confidence and confidence < 1, "Confidence must be in (0, 1)"
	f_dtype = _operator_checks(A)

	## Catch degenerate cases
	if (np.prod(A.shape) == 0) or (np.sum(A.shape) == 0):
		return 0

	## Parameter validation and sensible default settings for Lanczos
	rng = np.random.default_rng(seed)

	## Parameterize the convergence checking
	est = MeanEstimatorCLT(atol=atol, rtol=rtol) if stop == "confidence" else lambda x: False

	## All arguments validated and allocated, proceed with Monte-Carlo
	estimates = []  # trace estimates
	converged = False
	if fun is None:
		while not converged:
			N = A.shape[0]
			v = isotropic(size=(N, 1), seed=rng, method=pdf)
			tr_est = (v.T @ (A @ v)).item()
			estimates += [tr_est]
			converged = est(estimates[-1]) or len(estimates) >= maxiter
		return estimates
	else:
		## Reduce function should ufunc-like that accepts
		reduce = _reduce if reduce is None else reduce
		N, ncv, deg, orth, atol, rtol = _validate_lanczos(N=A.shape[0], ncv=ncv, deg=deg, orth=orth, atol=atol, rtol=rtol)

		## Allocate the tridiagonal elements + lanczos vectors in column-major storage
		alpha, beta = np.zeros(deg + 1, dtype=f_dtype), np.zeros(deg + 1, dtype=f_dtype)
		Q = np.zeros((N, ncv), dtype=f_dtype, order="F")

		## Parameterize the matrix function
		fun = param_callable(fun, **kwargs) if not isinstance(fun, Callable) else fun
		q_samples = []  # quadrature nodes and weights
		while not converged:
			v = isotropic(size=(N, 1), seed=rng, method=pdf)
			_lanczos.lanczos(A, v, deg, rtol, orth, alpha, beta, Q)
			## Todo: replace this with stemr calls to scipy Lapack
			nodes, weights = lanczos_quadrature(alpha, beta, deg=deg, quad="gw")
			# assert np.all(~np.isnan(nodes)) and np.all(~np.isnan(weights))
			# if np.all(~np.isnan(nodes)) and np.all(~np.isnan(weights)):
			nodes = fun(nodes)
			if info:
				q_samples += [(nodes, weights)]
			estimates += [N * np.sum(reduce(nodes, weights))]
			converged = est(estimates[-1]) or len(estimates) >= maxiter
		return np.fromiter(estimates, f_dtype) if not info else (np.fromiter(estimates, f_dtype), q_samples)

	# # hutch_args = (nv, distr_id, engine_id, seed, deg, 0.0, orth, ncv, quad_id, atol, rtol, num_threads, use_clt, t_values, z) # fmt: skip

	# ## Make the actual call
	# # info_dict = _trace.hutch(A, *hutch_args, **kwargs)
	# info_dict =

	# ## Return as early as possible if no additional info requested for speed
	# if not verbose and not info and not plot:
	# 	return info_dict["estimate"]

	# ## Post-process info dict
	# # std_err =
	# info_dict["estimator"] = "Girard-Hutchinson"
	# info_dict["valid"] = info_dict["samples"] != 0
	# info_dict["n_samples"] = np.sum(info_dict["valid"])
	# info_dict["n_matvecs"] = info_dict["n_samples"] * deg
	# info_dict["std_error"] = np.std(info_dict["samples"][info_dict["valid"]], ddof=1) / np.sqrt(info_dict["n_samples"])
	# info_dict["coeff_var"] = np.abs(info_dict["std_error"] / info_dict["estimate"])
	# info_dict["margin_of_error"] = (t_values[info_dict["n_samples"]] if info_dict["n_samples"] < 30 else z) * info_dict[
	# 	"std_error"
	# ]
	# info_dict["confidence"] = confidence
	# info_dict["stop"] = stop
	# info_dict["pdf"] = pdf
	# info_dict["rng"] = _engines[engine_id]
	# info_dict["seed"] = seed
	# info_dict["function"] = kwargs["function"]
	# info_dict["lanczos_kwargs"] = dict(orth=orth, ncv=ncv, deg=deg)
	# info_dict["quad"] = quad
	# info_dict["rtol"] = rtol
	# info_dict["atol"] = atol
	# info_dict["num_threads"] = "auto" if num_threads == 0 else num_threads
	# info_dict["maxiter"] = nv

	# ## Print the status if requested
	# if verbose:
	# 	print(_estimator_msg(info_dict))

	## Plot samples if requested
	# if plot:
	# 	from bokeh.plotting import show
	# 	from .plotting import figure_trace

	# 	p = figure_trace(info_dict["samples"])
	# 	show(p)
	# 	info_dict["figure"] = figure_trace(info_dict["samples"])

	## Final return
	# return (info_dict["estimate"], info_dict) if info else info_dict["estimate"]
