from typing import Callable, Optional, Protocol, Sized, Union, runtime_checkable

import numpy as np
import scipy as sp
from functools import partial

from .random import isotropic
from .estimators import ConvergenceEstimator, KneeEstimator, ToleranceEstimator, EstimatorResult
from .operators import _operator_checks


def diag(
	A: Union[sp.sparse.linalg.LinearOperator, np.ndarray],
	maxiter: int = 200,
	pdf: Union[str, Callable] = "rademacher",
	estimator: Union[str, ConvergenceEstimator] = "tolerance",
	seed: Union[int, np.random.Generator, None] = None,
	full: bool = False,
	callback: Optional[Callable] = None,
	**kwargs: dict,
) -> Union[float, tuple]:
	r"""Estimates the diagonal of a symmetric `A` via the Girard-Hutchinson estimator.

	This function uses up to `maxiter` random vectors to estimate of the diagonal of $A$ via the approximation:
	$$ \mathrm{diag}(A) = \sum_{i=1}^n e_i^T A e_i \approx n^{-1}\sum_{i=1}^n v^T A v $$
	When $v$ are isotropic, this approximation forms an unbiased estimator of the diagonal of $A$.

	:::{.callout-note}
	Convergence behavior is controlled by the `estimator` parameter: "confidence" uses the central limit theorem to generate confidence
	intervals on the fly, which may be used in conjunction with `atol` and `rtol` to upper-bound the error of the approximation.
	:::

	Parameters:
		A: real symmetric matrix or linear operator.
		maxiter: Maximum number of random vectors to sample for the trace estimate.
		pdf: Choice of zero-centered distribution to sample random vectors from.
		estimator: Type of estimator to use for convergence testing. See details.
		seed: Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
		full: Whether to return additional information about the computation.

	Returns:
		Estimate the diagonal of $A$. If `full = True`, additional information about the computation is also returned.

	See Also:
		- [lanczos](/reference/lanczos.lanczos.md): the lanczos tridiagonalization algorithm.
		- [ConfidenceEstimator](/reference/ConfidenceEstimator.md): Standard estimator of the mean from iid samples.

	Reference:
		1. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
		2. Hutchinson, Michael F. "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." Communications in Statistics-Simulation and Computation 18.3 (1989): 1059-1076.

	Examples:
		```{python}
		from primate.diagonal import diag
		```
	"""
	f_dtype = _operator_checks(A)
	N: int = A.shape[0]

	## Parameterize the random vector generation
	rng = np.random.default_rng(seed)
	pdf = isotropic(pdf=pdf, seed=rng)

	## Parameterize the convergence checking
	if isinstance(estimator, str):
		assert estimator in {"tolerance", "knee"}, "Only tolerance estimator is supported for now."
		if estimator == "tolerance":
			estimator = ToleranceEstimator(**{k: v for k, v in kwargs.items() if k in {"ord", "atol", "rtol"}})
		# elif estimator == "knee":
		# estimator = KneeEstimator(, transform=lambda x: )

	assert isinstance(estimator, ConvergenceEstimator), "`estimator` must satisfy the ConvergenceEstimator protocol."

	## Catch degenerate case
	if np.prod(A.shape) == 0:
		return 0.0 if not full else (0.0, EstimatorResult(0.0, False, "", 0, []))

	## Commence the Monte-Carlo iterations
	converged = False
	if full or callback is not None:
		numer, denom = np.zeros(N, dtype=f_dtype), np.zeros(N, dtype=f_dtype)
		result = EstimatorResult(numer, False, "", 0, [])
		while not converged:
			v = pdf(size=(N, 1), seed=rng).astype(f_dtype)
			u = (A @ v).ravel()
			numer += u * v.ravel()
			denom += np.square(v.ravel())
			estimator.update(np.atleast_2d(numer / denom))
			converged = estimator.converged() or len(estimator) >= maxiter
			result.update(estimator)
			if callback is not None:
				callback(result)
		return (estimator.estimate, result)
	else:
		numer, denom = np.zeros(N, dtype=f_dtype), np.zeros(N, dtype=f_dtype)
		while not converged:
			v = pdf(size=(N, 1), seed=rng).astype(f_dtype)
			u = (A @ v).ravel()
			numer += u * v.ravel()
			denom += np.square(v.ravel())
			estimator.update(np.atleast_2d(numer / denom))
			converged = estimator.converged() or len(estimator) >= maxiter
		return estimator.estimate


# def diagpp():
# 	pass


def xdiag(A: np.ndarray, m: int, pdf: str = "sphere", seed: Union[int, np.random.Generator, None] = None):
	"""Based on Program SM4.3, a MATLAB 2022b implementation for XDiag, by Ethan Epperly."""
	assert m >= 2, f"Number of matvecs must be at least 2."
	n, m = A.shape[0], m // 2
	# diag_prod = lambda A, B: np.diag(A.T @ B)[:, np.newaxis]
	diag_prod = lambda A, B: np.einsum("ij,ji->i", A.T, B)[:, np.newaxis]  # about 120-140% faster than above
	col_norm = lambda X: np.linalg.norm(X, axis=0)
	rng = np.random.default_rng(seed=seed)
	pdf = isotropic(pdf="sphere", seed=rng)

	## Sketching idea
	N = pdf(size=(n, m))
	Y = A @ N
	Q, R = np.linalg.qr(Y, mode="reduced")

	## Other quantities
	Z = A.T @ Q
	T = Z.T @ N
	R_inv = np.linalg.inv(R)
	S = R_inv / col_norm(R_inv)

	## Vector quantities
	dQZ, dQSSZ = diag_prod(Q.T, Z.T), diag_prod((Q @ S).T, (Z @ S).T)
	dNQT, dNY = diag_prod(N.T, (Q @ T).T), diag_prod(N.T, Y.T)
	dNQSST = diag_prod(N.T, (Q @ S @ np.diag(diag_prod(S, T).ravel())).T)

	## Diagonal estimate
	d = dQZ + (-dQSSZ + dNY - dNQT + dNQSST) / m
	return d.ravel()
