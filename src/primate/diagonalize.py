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
	"""Lanczos method of tridiagonalization.

	Description
	-----------
	This function implements the Lanczos method, or as Lanczos called it, the _method of minimized iterations_.

	Parameters
	----------
	A : LinearOperator | ndarray | sparray
	    Symmetric operator to tridiagonalize.
	v0 : ndarray, default = None
	    Initial vector to orthogonalize against.
	deg : int, default = None
	    Size of the Krylov subspace to expand.
	rtol : float, default = 1e-8
	    Relative tolerance to consider the invariant subspace as converged.
	orth : int, default = 0
	    Additional number of Lanczos vectors to orthogonalize against.
	sparse_mat : bool, default = False
	    Whether to output the diagonal and off-diagonal terms as a sparse matrix.
	return_basis : bool, default = False
	    Whether to return the `orth` + 2 Lanczos vectors.
	"""
	## Basic parameter validation
	n: int = A.shape[0]
	deg: int = A.shape[1] if deg is None else min(deg, A.shape[1])
	dt = dtype if dtype is not None else (A @ np.zeros(A.shape[1])).dtype
	assert deg > 0, "Number of steps must be positive!"

	## Determine number of projections + lanczos vectors
	orth: int = deg if orth < 0 or orth > deg else orth
	ncv: int = max(orth, 2) if not (return_basis) else n

	## Generate the starting vector if none is specified
	if v0 is None:
		rng = np.random.default_rng(seed)
		v0: np.ndarray = rng.uniform(size=A.shape[1], low=-1.0, high=+1.0).astype(dt)
	else:
		v0: np.ndarray = np.array(v0).astype(dt)
	assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."

	## Allocate the tridiagonal elements + lanczos vectors in column-major storage
	alpha, beta = np.zeros(deg + 1, dtype=np.float32), np.zeros(deg + 1, dtype=np.float32)
	Q = np.zeros((n, ncv), dtype=np.float32, order="F")
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
