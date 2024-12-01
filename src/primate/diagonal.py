from typing import Callable, Optional, Union

import numpy as np
import scipy as sp

from .random import isotropic
from .estimators import ConvergenceCriterion, convergence_criterion, MeanEstimator, EstimatorResult
from .operators import is_valid_operator


def diag(
	A: Union[sp.sparse.linalg.LinearOperator, np.ndarray],
	pdf: Union[str, Callable] = "rademacher",
	converge: Union[str, ConvergenceCriterion] = "tolerance",
	seed: Union[int, np.random.Generator, None] = None,
	full: bool = False,
	callback: Optional[Callable] = None,
	**kwargs: dict,
) -> Union[float, tuple]:
	r"""Estimates the diagonal of a symmetric `A` via the Girard-Hutchinson estimator.

	This function random vectors to estimate of the diagonal of $A$ via the approximation:
	$$ \mathrm{diag}(A) = \sum_{i=1}^n e_i^T A e_i \approx n^{-1}\sum_{i=1}^n v^T A v $$
	When $v$ are isotropic, this approximation forms an unbiased estimator of the diagonal of $A$.

	:::{.callout-note}
	Convergence behavior is controlled by the `estimator` parameter: "confidence" uses the central limit theorem to generate confidence
	intervals on the fly, which may be used in conjunction with `atol` and `rtol` to upper-bound the error of the approximation.
	:::

	Parameters:
		A: real symmetric matrix or linear operator.
		pdf: Choice of zero-centered distribution to sample random vectors from.
		estimator: Type of estimator to use for convergence testing. See details.
		seed: Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
		full: Whether to return additional information about the computation.

	Returns:
		Estimate the diagonal of $A$. If `full = True`, additional information about the computation is also returned.

	See Also:
		- [lanczos](/reference/lanczos.lanczos.md): the lanczos tridiagonalization algorithm.
		- [ConfidenceCriterion](/reference/ConfidenceCriterion.md): Standard estimator of the mean from iid samples.

	Reference:
		1. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
		2. Hutchinson, Michael F. "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." Communications in Statistics-Simulation and Computation 18.3 (1989): 1059-1076.

	Examples:
		```{python}
		from primate.diagonal import diag
		```
	"""
	f_dtype = is_valid_operator(A)
	N: int = A.shape[0]

	## Parameterize the random vector generation
	rng = np.random.default_rng(seed)
	pdf = isotropic(pdf=pdf, seed=rng)
	estimator = MeanEstimator(kwargs.pop("record", False))
	converge = convergence_criterion(converge, **kwargs)

	## Catch degenerate case
	if np.prod(A.shape) == 0:
		return 0.0 if not full else (0.0, EstimatorResult(0.0, False, "", 0, []))

	## Commence the Monte-Carlo iterations
	if full or callback is not None:
		numer, denom = np.zeros(N, dtype=f_dtype), np.zeros(N, dtype=f_dtype)
		result = EstimatorResult(numer, False, converge, 0, {})
		while not converge(estimator):
			v = pdf(size=(N, 1), seed=rng).astype(f_dtype)
			u = (A @ v).ravel()
			numer += u * v.ravel()
			denom += np.square(v.ravel())
			estimator.update(np.atleast_2d(numer / denom))
			result.update(estimator, converge)
			if callback is not None:
				callback(result)
		return (estimator.estimate, result)
	else:
		numer, denom = np.zeros(N, dtype=f_dtype), np.zeros(N, dtype=f_dtype)
		while not converge(estimator):
			v = pdf(size=(N, 1), seed=rng).astype(f_dtype)
			u = (A @ v).ravel()
			numer += u * v.ravel()
			denom += np.square(v.ravel())
			estimator.update(np.atleast_2d(numer / denom))
		return estimator.estimate


# def diagpp():
# 	pass


def xdiag(
	A: np.ndarray, m: Optional[int] = None, pdf: str = "sphere", seed: Union[int, np.random.Generator, None] = None
):
	"""Estimates the diagonal of `A` using `m / 2` matrix-vector multiplications.

	Based originally on Program SM4.3, a MATLAB 2022b implementation for XDiag, by Ethan Epperly.
	"""
	m = 2 * A.shape[0] if m is None else min(m + (m % 2), 2 * A.shape[0])
	n, m = A.shape[0], m // 2

	## Configure
	diag_prod = lambda A, B: np.einsum("ij,ji->i", A.T, B)[:, np.newaxis]  # about 120-140% faster than np.diag(A.T @ B)
	rng = np.random.default_rng(seed=seed)
	pdf = isotropic(pdf=pdf, seed=rng)

	## Sketching idea
	N = pdf(size=(n, m))
	Y = A @ N
	Q, R = np.linalg.qr(Y, mode="reduced")
	dNY = diag_prod(N.T, Y.T)
	del Y

	## Matrix quantities
	## TODO: see if dtrtri compiles on all platforms, https://stackoverflow.com/questions/6042308/numpy-inverting-an-upper-triangular-matrix
	Z = A.T @ Q
	T = Z.T @ N
	R_inv = np.linalg.inv(R)
	S = R_inv / np.linalg.norm(R_inv, axis=0)
	QS = Q @ S

	## Vector quantities
	dQZ = diag_prod(Q.T, Z.T)
	dQSSZ = diag_prod(QS.T, (Z @ S).T)
	dNTQ = diag_prod(N.T, (Q @ T).T)
	dNQSST = diag_prod(N.T, (diag_prod(S, T) * QS.T))
	# dNQSST = diag_prod(N.T, (Q @ S @ np.diag(diag_prod(S, T).ravel())).T)

	## Diagonal estimate
	d = dQZ + (-dQSSZ + dNY - dNTQ + dNQSST) / m
	return d.ravel()
