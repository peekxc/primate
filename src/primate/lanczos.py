from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse import sparray, spdiags
from scipy.sparse.linalg import LinearOperator, eigsh

from . import _lanczos
from .fttr import ortho_poly
from .tridiag import eigh_tridiag, eigvalsh_tridiag


def _validate_lanczos(N: int, ncv: int, deg: int, orth: int, atol: float, rtol: float) -> tuple:
	deg: int = N if deg < 0 else int(np.clip(deg, 1, N))  # 1 <= deg <= N
	ncv: int = int(np.clip(ncv, 2, min(deg, N)))  # 2 <= ncv <= deg
	orth: int = int(max(0, np.clip(orth, 0, ncv - 2)))  # orth <= (deg - 2)
	atol: float = 0.0 if atol is None else float(atol) / N  # adjust for quadrature est.
	rtol: float = 0.0 if rtol is None else float(rtol)
	assert (
		ncv >= 2 and ncv >= (orth + 2) and ncv <= deg
	), f"Invalid Lanczos parameters; (orth < ncv? {orth < ncv}, ncv >= 2 ? {ncv >= 2}, ncv <= deg? {ncv <= deg})"
	assert not np.isnan(atol), "Absolute tolerance is NAN!"
	return N, ncv, deg, orth, atol, rtol


def lanczos(
	A: Union[np.ndarray, sparray, LinearOperator],
	v0: Optional[np.ndarray] = None,
	deg: Optional[int] = None,
	rtol: float = 1e-8,
	orth: int = 0,
	sparse_mat: bool = False,
	return_basis: bool = False,
	seed: Union[int, np.random.Generator, None] = None,
	dtype: Optional[np.dtype] = None,
	**kwargs,
) -> tuple:
	"""Lanczos method for symmetric tridiagonalization.

	This function implements Paiges A27 variant (1) of the Lanczos method for tridiagonalizing linear operators, with additional
	modifications to support varying degrees of re-orthogonalization. In particular, `orth=0` corresponds to no re-orthogonalization,
	`orth < deg` corresponds to partial re-orthogonalization, and `orth >= deg` corresponds to full re-orthogonalization.

	The Lanczos method builds a tridiagonal `T` from a symmetric `A` via an orthogonal change-of-basis `Q`:
	$$ Q^T A Q  = T $$
	Unlike other Lanczos implementations (e.g. SciPy's `eigsh`), which includes e.g. sophisticated restarting,
	deflation, and selective-reorthogonalization steps, this method simply executes `deg` steps of the Lanczos method with
	the supplied `v0` and returns the diagonals of the resulting tridiagonal matrix `T`.

	Rayleigh-Ritz approximations of the eigenvalues of `A` can be further obtained by diagonalizing `T` via any
	symmetric tridiagonal eigenvalue solver, `scipy.linalg.eigh_tridiagonal` though note unlike `eigsh` no checking is performed
	for 'ghost' or already converged eigenvalues. To increase the accuracy of these eigenvalue approximation, try increasing `orth`
	and `deg`. Supplying either negative values or values larger than `deg` for `orth` will result in full re-orthogonalization,
	though note the number of matvecs scales linearly with `deg` and the number of inner-products scales quadratically with `orth`.

	Parameters:
		A: Symmetric operator to tridiagonalize.
		v0: Initial vector to orthogonalize against.
		deg: Size of the Krylov subspace to expand.
		rtol: Relative tolerance to consider the invariant subspace as converged.
		orth: Number of additional Lanczos vectors to orthogonalize against.
		sparse_mat: Whether to output the tridiagonal matrix as a sparse matrix.
		return_basis: If `True`, returns the Krylov basis vectors `Q`.
		dtype: The precision dtype to specialize the computation.

	Returns:
		A tuple `(a,b)` parameterizing the diagonal and off-diagonal of the tridiagonal Jacobi matrix. If `return_basis=True`,
		the tuple `(a,b), Q` is returned, where `Q` represents an orthogonal basis for the degree-`deg` Krylov subspace.

	See Also:
		- scipy.linalg.eigh_tridiagonal : Eigenvalue solver for real symmetric tridiagonal matrices.
		- operator.matrix_function : Approximates the action of a matrix function via the Lanczos method.

	References:
		1. Paige, Christopher C. "Computational variants of the Lanczos method for the eigenproblem." IMA Journal of Applied Mathematics 10.3 (1972): 373-381.
	"""
	## Basic parameter validation
	n: int = A.shape[0]
	deg: int = A.shape[1] if deg is None else min(deg, A.shape[1])
	dt = dtype if dtype is not None else (A @ np.zeros(A.shape[1])).dtype
	assert deg > 0, "Number of steps must be positive!"

	## Get the dtype; infer it if it's not available
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert f_dtype.type in {np.float32, np.float64}, "Only 32- or 64-bit floating point numbers are supported."

	## Determine number of projections + lanczos vectors
	orth: int = deg if orth < 0 or orth > deg else orth
	ncv: int = np.clip(orth, 2, deg) if not (return_basis) else deg
	# _validate_lanczos(n, ncv, deg, orth)

	## Generate the starting vector if none is specified
	if v0 is None:
		rng = np.random.default_rng(seed)
		v0: np.ndarray = rng.uniform(size=A.shape[1], low=-1.0, high=+1.0).astype(dt)
	else:
		v0: np.ndarray = np.array(v0).astype(dt)
	assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."

	## Allocate the tridiagonal elements + lanczos vectors in column-major storage
	alpha = kwargs.get("alpha", np.zeros(deg + 1, dtype=f_dtype))
	beta = kwargs.get("beta", np.zeros(deg + 1, dtype=f_dtype))
	Q = kwargs.get("Q", np.zeros((n, ncv), dtype=f_dtype, order="F"))
	assert isinstance(alpha, np.ndarray) and len(alpha) == deg + 1 and alpha.dtype == f_dtype and alpha.flags["WRITEABLE"]
	assert isinstance(beta, np.ndarray) and len(beta) == deg + 1 and beta.dtype == f_dtype and beta.flags["WRITEABLE"]
	assert Q.ndim == 2 and Q.shape == (n, ncv) and Q.flags["F_CONTIGUOUS"] and Q.flags["WRITEABLE"] and Q.flags["OWNDATA"]

	## Call the procedure
	_lanczos.lanczos(A, v0, deg, rtol, orth, alpha, beta, Q)

	## Format the output(s)
	if sparse_mat:
		T = spdiags(data=[np.roll(beta, -1), alpha, beta], diags=(-1, 0, +1), m=deg, n=deg)
		return T if not return_basis else (T, Q)
	else:
		a, b = alpha[:deg], beta[1:deg]
		return (a, b) if not return_basis else ((a, b), Q)


def rayleigh_ritz(A, deg: Optional[int] = None, return_eigenvectors: bool = False, method: str = "RRR", **kwargs: dict):
	n: int = A.shape[0]
	deg: int = A.shape[1] if deg is None else min(deg, A.shape[1])
	assert deg > 0, "Number of steps must be positive!"

	## Run the lanczos method
	deg = np.clip([deg], 2, n).item()
	Q_basis = kwargs.pop("return_basis", False)
	if Q_basis:
		(a, b), Q = lanczos(A, deg=deg, return_basis=True, **kwargs)
	else:
		(a, b) = lanczos(A, deg=deg, return_basis=False, **kwargs)

	## Return Rayleigh-Ritz values, Ritz vectors Y (if requested), and Lanczos basis Q (if requested)
	if return_eigenvectors:
		rw, Y = eigh_tridiag(a, b)
		return (rw, Y) if not Q_basis else (rw, Y, Q)
	else:
		rw = eigvalsh_tridiag(a, b)
		return rw if not Q_basis else (rw, Q)


## TODO: block lanczos + ABLE method
## https://netlib.org/utk/people/JackDongarra/etemplates/node250.html
## Also: Chen's Krylov-aware method


def _lanczos_py(A: np.ndarray, v0: np.ndarray, k: int, tol: float) -> int:  # pragma: no cover
	"""Base lanczos algorithm, for establishing a baseline"""
	n = A.shape[0]
	assert k <= n, "Can perform at most k = n iterations"
	assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."
	alpha = np.zeros(n + 1, dtype=np.float32)
	beta = np.zeros(n + 1, dtype=np.float32)
	qp = np.zeros(n, dtype=np.float32)
	qc = v0.copy()
	qc /= np.linalg.norm(v0)
	for i in range(k):
		qn = A @ qc - beta[i] * qp
		alpha[i] = np.dot(qn, qc)
		qn -= alpha[i] * qc
		beta[i + 1] = np.linalg.norm(qn)
		if np.isclose(beta[i + 1], tol):
			break
		qn /= beta[i + 1]
		qp, qc = qc, qn
	return alpha, beta


def _orth_vector(v, U, start_idx, p, reverse=False):  # pragma: no cover
	n = U.shape[0]
	m = U.shape[1]
	tol = 2 * np.finfo(U.dtype).eps * np.sqrt(n)

	diff = -1 if reverse else 1
	for c in range(p):
		i = (start_idx + c * diff) % m
		u_norm = np.linalg.norm(U[:, i]) ** 2
		s_proj = np.dot(v, U[:, i])
		if u_norm > tol and np.abs(s_proj) > tol:
			v -= (s_proj / u_norm) * U[:, i]


def _lanczos_recurrence(A, q, deg, rtol, orth, V, ncv):  # pragma: no cover
	n, m = A.shape
	residual_tol = np.sqrt(n) * rtol

	Q = np.zeros((n, ncv), dtype=A.dtype)
	v = q.copy()
	Q[:, 0] = v / np.linalg.norm(v)
	beta = np.zeros(deg + 1, dtype=A.dtype)
	alpha = np.zeros(deg, dtype=A.dtype)
	pos = [ncv - 1, 0, 1]

	for j in range(deg):
		v = A @ Q[:, pos[1]]
		v -= beta[j] * Q[:, pos[0]]
		alpha[j] = np.dot(Q[:, pos[1]], v)
		v -= alpha[j] * Q[:, pos[1]]

		if orth > 0:
			_orth_vector(v, Q, pos[1], orth, reverse=True)

		beta[j + 1] = np.linalg.norm(v)
		if beta[j + 1] < residual_tol or (j + 1) == deg:
			break

		Q[:, pos[2]] = v / beta[j + 1]
		pos = [pos[1], pos[2], (pos[2] + 1) % ncv]

	return alpha, beta[:-1], Q


from numpy.polynomial import Polynomial


class OrthogonalPolynomialBasis(Polynomial):
	def __init__(self, A, deg: int) -> None:
		self.a, self.b = lanczos(A, deg=deg)
		self.theta, self.rv = eigh_tridiag(self.a, self.b)
		self.tau = np.square(self.rv[0, :])
		self.mu_0 = np.sum(np.abs(self.theta[:deg]))
		self.mu_sqrt_rec = 1.0 / np.sqrt(self.mu_0)
		self.dtype = A.dtype

	def fit(self, x: Union[np.ndarray, int], y: np.ndarray):
		deg = len(self.a)
		self.Z_ = np.zeros((len(x), deg), dtype=self.dtype)
		for i, xi in enumerate(x):
			ortho_poly(xi, self.mu_sqrt_rec, self.a, self.b, self.Z_[i, :], deg)
		self.coef = np.linalg.solve(self.Z_.T @ self.Z_, self.Z_.T @ y)
		return self.Z_ @ self.coef

	def __call__(self, x: np.ndarray):
		x = np.atleast_1d(x)
		deg = len(self.a)
		Z = np.zeros((len(x), deg), dtype=self.dtype)
		for i, xi in enumerate(x):
			ortho_poly(xi, self.mu_sqrt_rec, self.a, self.b, Z[i, :], deg)
		return Z @ self.coef
