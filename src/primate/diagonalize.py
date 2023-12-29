import numpy as np
from typing import Optional
from scipy.sparse import spdiags
from scipy.sparse.linalg import LinearOperator
import _lanczos


def lanczos(
	A: LinearOperator,
	v0: Optional[np.ndarray] = None,
	deg: int = None,
	rtol: float = 1e-8,
	orth: int = 0,
	sparse_mat: bool = False,
	return_basis: bool = False,
	seed: int = None,
	dtype=None,
) -> tuple:
	"""Lanczos method for matrix tridiagonalization.

	This function implements Paiges A27 variant (1) of the Lanczos method for tridiagonalizing linear operators, with additional 
	modifications to support varying degrees of re-orthogonalization. In particular, `orth=0` corresponds to no re-orthogonalization, 
	`orth < deg` corresponds to partial re-orthogonalization, and `orth >= deg` corresponds to full re-orthogonalization.
	
	Notes
	-----
	The Lanczos method builds a tridiagonal `T` from a symmetric `A` via an orthogonal change-of-basis `Q`:
	$$ Q^T A Q  = T $$
	Unlike other Lanczos implementations (e.g. SciPy's `eigsh`), which includes e.g. sophisticated restarting, 
	deflation, and selective-reorthogonalization steps, this method simply executes `deg` steps of the Lanczos method with 
	the supplied `v0` and returns the resulting tridiagonal matrix `T`.
	
	Rayleigh-Ritz approximations of the eigenvalues of `A` can be further obtained by diagonalizing `T` via any 
	symmetric tridiagonal eigenvalue solver, `scipy.linalg.eigh_tridiagonal` though note unlike `eigsh` no checking is performed 
	for 'ghost' or already converged eigenvalues. To increase the accuracy of these eigenvalue approximation, try increasing `orth` 
	and `deg`. Supplying either negative values or values larger than `deg` for `orth` will result in full re-orthogonalization, 
	though note the number of matvecs scales linearly with `deg` and the number of inner-products scales quadratically with `orth`.

	Parameters
	----------
	A : LinearOperator or ndarray or sparray
	    Symmetric operator to tridiagonalize.
	v0 : ndarray, default=None
	    Initial vector to orthogonalize against.
	deg : int, default=None
	    Size of the Krylov subspace to expand.
	rtol : float, default=1e-8
	    Relative tolerance to consider the invariant subspace as converged.
	orth : int, default=0
	    Number of additional Lanczos vectors to orthogonalize against.
	sparse_mat : bool, default=False
	    Whether to output the tridiagonal matrix as a sparse matrix.
	return_basis : bool, default=False
	    If `True`, returns the Krylov basis vectors `Q`.
	dtype : dtype, default=None
	  	The precision dtype to specialize the computation.

	Returns
	-------
	: 
			A tuple `(a,b)` parameterizing the diagonal and off-diagonal of the tridiagonal matrix. If `return_basis=True`, 
			the tuple `(a,b), Q` is returned, where `Q` represents an orthogonal basis for the degree-`deg` Krylov subspace.

	See Also
	--------
	scipy.linalg.eigh_tridiagonal : Eigenvalue solver for real symmetric tridiagonal matrices.
	operator.matrix_function : Approximates the action of a matrix function via the Lanczos method.

	References
	----------
	1. Paige, Christopher C. "Computational variants of the Lanczos method for the eigenproblem." IMA Journal of Applied Mathematics 10.3 (1972): 373-381.
	"""
	## Basic parameter validation
	n: int = A.shape[0]
	deg: int = A.shape[1] if deg is None else min(deg, A.shape[1])
	dt = dtype if dtype is not None else (A @ np.zeros(A.shape[1])).dtype
	assert deg > 0, "Number of steps must be positive!"

	## Get the dtype; infer it if it's not available
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert (
		f_dtype.type == np.float32 or f_dtype.type == np.float64
	), "Only 32- or 64-bit floating point numbers are supported."

	## Determine number of projections + lanczos vectors
	orth: int = deg if orth < 0 or orth > deg else orth
	ncv: int = np.clip(orth, 2, deg) if not (return_basis) else n

	## Generate the starting vector if none is specified
	if v0 is None:
		rng = np.random.default_rng(seed)
		v0: np.ndarray = rng.uniform(size=A.shape[1], low=-1.0, high=+1.0).astype(dt)
	else:
		v0: np.ndarray = np.array(v0).astype(dt)
	assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."

	## Allocate the tridiagonal elements + lanczos vectors in column-major storage
	alpha, beta = np.zeros(deg + 1, dtype=f_dtype), np.zeros(deg + 1, dtype=f_dtype)
	Q = np.zeros((n, ncv), dtype=f_dtype, order="F")
	assert Q.flags["F_CONTIGUOUS"] and Q.flags["WRITEABLE"] and Q.flags["OWNDATA"]

	## Call the procedure
	_lanczos.lanczos(A, v0, deg, rtol, orth, alpha, beta, Q)

	## Format the output(s)
	if sparse_mat:
		T = spdiags(data=[beta, alpha, np.roll(beta, 1)], diags=(-1, 0, +1), m=deg, n=deg)
		return T if not return_basis else (T, Q)
	else:
		a, b = alpha[:deg], beta[1:deg]
		return (a, b) if not return_basis else ((a, b), Q)
