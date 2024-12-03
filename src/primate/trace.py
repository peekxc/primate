"""Estimators involving matrix function, trace, and diagonal estimation."""

from itertools import islice
from typing import Callable, Generator, Iterable, Optional, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .estimators import (
	ConfidenceCriterion,
	ConvergenceCriterion,
	CountCriterion,
	EstimatorResult,
	MeanEstimator,
	convergence_criterion,
)
from .linalg import update_trinv
from .operators import is_valid_operator
from .random import isotropic


## TODO: should return views when possible
def _chunk(iterable: Iterable, n: int) -> Generator:
	"""Numpy-aware chunking, which yield successive n-sized chunks as either views or tuples from an iterable  or ndarray."""
	assert n >= 1, "n must be at least one"
	if isinstance(iterable, np.ndarray):
		yield from np.array_split(iterable, 3)
	else:
		iterator = iter(iterable)
		while batch := tuple(islice(iterator, n)):
			yield batch


def hutch(
	A: Union[LinearOperator, np.ndarray],
	batch: int = 32,
	pdf: Union[str, Callable] = "rademacher",
	converge: Union[str, ConvergenceCriterion] = "default",
	seed: Union[int, np.random.Generator, None] = None,
	full: bool = False,
	callback: Optional[Callable] = None,
	**kwargs: dict,
) -> Union[float, tuple]:
	r"""Estimates the trace of a symmetric `A` via the Girard-Hutchinson estimator.

	This function uses up to `maxiter` random vectors to estimate of the trace of $A$ via the approximation:
	$$ \mathrm{tr}(A) = \sum_{i=1}^n e_i^T A e_i \approx n^{-1}\sum_{i=1}^n v^T A v $$
	When $v$ are isotropic, this approximation forms an unbiased estimator of the trace.

	:::{.callout-note}
	Convergence behavior is controlled by the `estimator` parameter: "confidence" uses the central limit theorem to generate confidence
	intervals on the fly, which may be used in conjunction with `atol` and `rtol` to upper-bound the error of the approximation.
	:::

	Parameters:
		A: real symmetric matrix or linear operator.
		batch: Number of random vectors to sample at a time for batched matrix multiplication.
		pdf: Choice of zero-centered distribution to sample random vectors from.
		converge: Convergence criterion to test for estimator convergence. See details.
		seed: Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
		full: Whether to return additional information about the computation.
		callback: Optional callable to execute after each batch of samples.
		**kwargs: Additional keyword arguments to parameterize the convergence criterion.

	Returns:
		Estimate the trace of $f(A)$. If `info = True`, additional information about the computation is also returned.

	See Also:
		- [lanczos](/reference/lanczos.lanczos.md): the lanczos tridiagonalization algorithm.
		- [MeanEstimator](/reference/MeanEstimator.md): Standard estimator of the mean from iid samples.
		- [ConfidenceCriterion](/reference/ConfidenceCriterion.md): Criterion for convergence that uses the central limit theorem.

	Reference:
		1. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
		2. Hutchinson, Michael F. "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." Communications in Statistics-Simulation and Computation 18.3 (1989): 1059-1076.

	Examples:
		```{python}
		from primate.trace import hutch
		```
	"""
	f_dtype = is_valid_operator(A)
	N: int = A.shape[0]

	## Parameterize the various quantities
	rng = np.random.default_rng(seed)
	pdf = isotropic(pdf=pdf, seed=rng)
	estimator = MeanEstimator(record=kwargs.pop("record", False))
	if converge == "default":
		cc1 = CountCriterion(count=200)
		cc2 = ConfidenceCriterion(confidence=0.95, atol=1.0, rtol=0.0)
		converge = cc1 | cc2
	else:
		converge = convergence_criterion(converge, **kwargs)
	# quad_form = (lambda v: A.quad(v)) if hasattr(A, "quad") else (lambda v: (v.T @ (A @ v)).item())
	quad_form = (lambda v: A.quad(v)) if hasattr(A, "quad") else (lambda v: np.diag(np.atleast_2d((v.T @ (A @ v)))))

	## Catch degenerate case
	if np.prod(A.shape) == 0:
		return 0.0 if not full else (0.0, EstimatorResult(estimator, converge))

	## Commence the Monte-Carlo iterations
	if full or callback is not None:
		result = EstimatorResult(estimator, converge)
		callback = (lambda x: x) if callback is None else callback
		while not converge(estimator):
			v = pdf(size=(N, batch)).astype(f_dtype)
			estimator.update(quad_form(v))
			result.update(estimator, converge)
			callback(result)
		return (estimator.estimate, result)
	else:
		while not converge(estimator):
			estimator.update(quad_form(pdf(size=(N, 1)).astype(f_dtype)))
		return estimator.estimate


def hutchpp(
	A: Union[LinearOperator, np.ndarray],
	m: Optional[int] = None,
	batch: int = 32,
	mode: str = "reduced",
	pdf: Union[str, Callable] = "rademacher",
	seed: Union[int, np.random.Generator, None] = None,
	full: bool = False,
) -> Union[float, dict]:
	"""Hutch++ estimator.

	Parameters:
		A: Matrix or LinearOperator to estimate the trace of.
		m: number of matvecs to use. If not given, defaults to `n // 3`.
		batch: currently unused.
	"""
	f_dtype = is_valid_operator(A)
	N: int = A.shape[0]

	## Parameterize the random vector generation
	rng = np.random.default_rng(seed)
	pdf = isotropic(pdf=pdf, seed=rng)

	## Prepare quadratic form evaluator
	quad_form = (lambda v: A.quad(v)) if hasattr(A, "quad") else (lambda v: (v.T @ (A @ v)).item())

	## Catch degenerate case
	if np.prod(A.shape) == 0:
		return 0.0 if not full else (0.0, EstimatorResult(0.0, False, "", 0, []))

	## Setup constants
	nb = (N // 3) if m is None else m  # number of samples to dedicate to deflation
	nb += nb % 3  # ensure nb divides by 3
	# maxiter: int = (N // 3) if maxiter == "auto" else int(maxiter)  # residual samples; default rule uses Hutch++ result
	# assert nb % 3 == 0, "Number of samples must be divisible by 3"

	## Sketch Y / Q - use numpy for now, but consider parallelizing MGS later
	WB = pdf(size=(N, nb)).astype(f_dtype)
	Q = np.linalg.qr(A @ WB, mode="reduced")[0]

	## Estimate trace of the low-rank sketch
	## Full mode may not be space efficient, but is potentially vectorized, so suitable for relatively small output dimen.
	## Uses at most O(n) memory, but potentially slower
	## https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v
	rng_ests = np.einsum("...i,...i->...", A @ Q, Q) if mode == "full" else np.array([quad_form(q) for q in Q.T])
	tr_rng = np.sum(rng_ests)

	## Estimate trace of the residual on the deflated subspaces
	G = pdf(size=(N, nb), seed=rng).astype(f_dtype)
	G -= Q @ (Q.T @ G)
	defl_ests = np.einsum("...i,...i->...", A @ G, G)  # [(g @ A @ g) for g in G[g_1, g_2, ..., g_nb]]
	tr_defl = (1 / nb) * np.sum(defl_ests)

	if not full:
		return tr_rng + tr_defl
	else:
		result = EstimatorResult(0.0, False, None, 0, {})
		result.estimate = tr_rng + tr_defl
		result.nit = 2 * nb
		result.samples = np.concatenate([rng_ests, defl_ests])
		return result.estimate, result


def _xtrace(W: np.ndarray, Z: np.ndarray, Q: np.ndarray, R: np.ndarray, R_inv: np.ndarray, pdf: str):
	"""Helper for xtrace function.

	Parameters:
		W: all isotropic random vectors sampled thus far.
		Z: the image A @ Q.
		Q: orthogonal component of qr(A @ W)
		R: upper-triangular component of qr(A @ W)
		R_inv: inverse matrix of R.
		pdf: the distribution with which `W` was sampled from.

	Returns:
		tuple (t, est, err) representing the averaged leave-one-out trace estimate
	"""
	# diag_prod = lambda A, B: np.diag(A.T @ B)[:, np.newaxis]
	diag_prod = lambda A, B: np.einsum("ij,ji->i", A.T, B)[:, np.newaxis]  ## Faster version of the above

	n, m = W.shape
	W_proj = Q.T @ W
	# R_inv = solve_triangular(R, np.identity(m)).T  # todo: replace with dtrtri?
	S = R_inv / np.linalg.norm(R_inv, axis=0)  # S == 'spherical', makes columns unit norm

	## Handle the scale
	if not pdf == "sphere":
		scale = np.ones(m)[:, np.newaxis]  # this is a column vector
	else:
		col_norm = lambda X: np.linalg.norm(X, axis=0)
		c = n - m + 1
		scale = c / (n - (col_norm(W_proj)[:, np.newaxis]) ** 2 + (diag_prod(S, W_proj) * col_norm(S)[:, np.newaxis]) ** 2)

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
	return tr_ests
	# t = tr_ests.mean()
	# err = np.std(tr_ests) / np.sqrt(m)
	# return t, tr_ests, err


def xtrace(
	A: Union[LinearOperator, np.ndarray],
	batch: int = 32,
	pdf: Union[str, Callable] = "sphere",
	converge: Union[str, ConvergenceCriterion] = "default",
	seed: Union[int, np.random.Generator, None] = None,
	full: bool = False,
	callback: Optional[Callable] = None,
	**kwargs: dict,
) -> Union[float, tuple]:
	"""Estimates the trace of `A` using the XTrace estimator.

	This function implements Epperly's exchangeable 'XTrace' estimator.

	Parameters:
		A: real symmetric matrix or linear operator.
		batch: Number of random vectors to sample at a time for batched matrix multiplication.
		pdf: Choice of zero-centered distribution to sample random vectors from.
		converge: Convergence criterion to test for estimator convergence. See details.
		seed: Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
		full: Whether to return additional information about the computation.
		callback: Optional callable to execute after each batch of samples.
		**kwargs: Additional keyword arguments to parameterize the convergence criterion.

	Returns:
		Estimate the trace of `A`. If `info = True`, additional information about the computation is also returned.
	"""
	from scipy.linalg import qr_insert

	assert batch >= 1, "Batch size must be positive."
	# assert atol >= 0.0 and rtol >= 0.0, "Error tolerances must be positive"
	# assert cond_tol >= 0.0, "Condition number must be non-negative"
	n = A.shape[0]
	callback = (lambda result: ...) if not callable(callback) else callback
	record = kwargs.pop("record", False)
	estimator = MeanEstimator(record=record)

	## Parameterize the convergence criteria
	if converge == "default":
		cc1 = CountCriterion(count=n)
		cc2 = ConfidenceCriterion(confidence=0.95)
		converge = cc1 | cc2
	else:
		converge = CountCriterion(count=n)
		converge |= convergence_criterion(converge, **kwargs)

	## Setup outputs. TODO: these should really be resizable arrays
	W = np.zeros(shape=(n, 0))  # Isotropic vectors
	Y = np.zeros(shape=(n, 0))  # Im(A @ W)
	Z = np.zeros(shape=(n, 0))  # Im(A @ orth(Y))
	Q, R = np.linalg.qr(Y, mode="reduced")
	R_inv = np.zeros(shape=(0, 0))

	## Commence the batch-iterations
	result = EstimatorResult()
	rng = np.random.default_rng(seed)
	while not converge(estimator):
		## Determine number of new sample vectors to generate
		ns = min(A.shape[1] - W.shape[1], int(batch))

		## Sample a batch of random isotropic vectors
		## TODO: replace with proper batch updates
		## TODO: https://stackoverflow.com/questions/6042308/numpy-inverting-an-upper-triangular-matrix
		N = isotropic(size=(ns, n), pdf=pdf, seed=rng)  # 'new' vectors
		for eta in N:
			y = A @ eta.T
			Q, R = qr_insert(Q, R, u=y, k=Q.shape[1], which="col")  # rcond=FLOAT_MIN
			R_inv = update_trinv(R_inv, R[:, -1])
		W = np.c_[W, N.T]
		Z = np.c_[Z, A @ Q[:, -ns:]]

		## Expand the subspace
		t_samples = _xtrace(W, Z, Q, R, R_inv, pdf)

		## Test for convergence
		estimator = MeanEstimator(record=record)  # degenerate approach since XTrace tracks this
		estimator.update(t_samples.ravel())
		result.update(estimator, converge)
		callback(result)

	return (result.estimate, result) if full else result.estimate
