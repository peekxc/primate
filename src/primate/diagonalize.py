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

	This function implements Paige's A27 variant[1]_ of the Lanczos method.

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
	orth : int, default=0
	    Additional number of Lanczos vectors to orthogonalize against.
	sparse_mat : bool, default = False
	    Whether to output the diagonal and off-diagonal terms as a sparse matrix.
	return_basis : bool, default = False
	    Whether to return the `orth` + 2 Lanczos vectors.
	dtype : dtype, default = None
	                The precision dtype to specialize the computation.

	Returns
	-------
	(a,b) : tuple
	    The diagonal and off-diagonal of the tridiagonal matrix.
	Q : ndarray
	                If `return_basis` is True, the orthogonal basis for the Krylov subspace.

	See Also
	--------
	scipy.linalg.eigh_tridiagonal : Eigenvalue solver for real symmetric tridiagonal matrices.
	operator.matrix_function : Approximates the action of a matrix function via the Lanczos method.

	Notes
	-----
	No checking for ghost- or otherwise degenerate eigenvalues is performed. To increase the accuracy of the eigenvalue approximation, increase `orth` and `deg`,
	Note the complexity of the iteration scales linearly with `deg` and quadratically with `orth`.

	Supplying either negative values or values larger than `deg` for `orth` will result in full re-orthogonalization.

	.. [1] Paige, Christopher C. "Computational variants of the Lanczos method for the eigenproblem." IMA Journal of Applied Mathematics 10.3 (1972): 373-381.

	Examples
	----------
	```{python}
	import numpy as np 
	from scipy.linalg import eigh_tridiagonal
	from primate.diagonalize import lanczos
  
	## Generate a random symmetric matrix
  A = np.random.normal(size=(100,100))
	A = (A + A.T) / 2

	## Perform the Lanczos expansion on the Krylov subspace (A, v0)
  v0 = np.random.uniform(size=n)
  (a,b) = lanczos(A, v0)
  
	## Rayleigh-Ritz approximations via eigh_tridiagonal
	rr = eigh_tridiagonal(a, b, eigvals_only=True)

	## Compare with true eigenvalues
	ew = np.linalg.eigh(A)[0]
	assert npp.allclose(rr, ew, atol=1e-3)
	```
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
	ncv: int = max(orth, 2) if not (return_basis) else n

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
