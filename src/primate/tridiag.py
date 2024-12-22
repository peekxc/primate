import numpy as np
from .tqli import tqli
from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal


## TODO: add banded solver
def _eigh_tridiag(d: np.ndarray, e: np.ndarray, Z: np.ndarray, method: str = "auto", maxiter: int = 30):
	# assert len(d) == len(e), "Main diagonal and subdiagonal lengths not equal."
	if method == "mrrr":
		eigh_solver = eigvalsh_tridiagonal if np.prod(Z.shape) == 0 else eigh_tridiagonal
		return eigh_solver(d, e[1:])
	elif method == "tqli":
		tqli(d, e, Z, maxiter)
		return d if np.prod(Z.shape) == 0 else (d, Z)
	else:
		try:
			res = _eigh_tridiag(d, e, Z, method="mrrr")
			return res
		except np.linalg.LinAlgError:
			return _eigh_tridiag(d, e, Z, method="tqli")


## Note this issue: https://github.com/scipy/scipy/issues/13982
## Seems banded is slightly more efficient for full decomposition
def eigh_tridiag(d: np.ndarray, e: np.ndarray, method: str = "auto", maxiter: int = 30):
	"""Finds the eigenpairs of a symmetric real tridiagonal matrix.

	Parameters:
		d: main diagonal of length n.
		e: subdiagonal of length n. First element must be zero.
		Z: output matrix to store eigenvectors.
		method: one of 'mrrr', 'tqli', or 'auto'. Defaults to 'auto'.
		maxiter: the number of iterations for the tqli algorithm. Defaults to 30.

	Returns:
		Ritz pairs `(r, Y)` where `r` represents the rayleigh-ritz values and `Y` their corresponding Ritz vectors.
	"""
	assert method in {"tqli", "mrrr", "auto"}
	assert len(d) in {len(e) + 1, len(e)}, "Invalid diagonal/subdiagonal pair"
	e = np.append([0], e) if len(e) == (len(d) - 1) else e
	d, e = (d.copy(), e.copy())
	Z = np.eye(len(d), dtype=d.dtype)
	return _eigh_tridiag(d, e, Z, method, maxiter)


def eigvalsh_tridiag(d: np.ndarray, e: np.ndarray, method: str = "auto", maxiter: int = 30):
	"""Finds the eigenvalues of a symmetric real tridiagonal matrix.

	Parameters:
		d: main diagonal of length n.
		e: subdiagonal of length n. First element must be zero.
		method: one of 'mrrr', 'tqli', or 'auto'. Defaults to 'auto'.

	Returns:
		Ritz pairs `(r, Y)` where `r` represents the rayleigh-ritz values and `Y` their corresponding Ritz vectors.
	"""
	assert method in {"tqli", "mrrr", "auto"}
	assert len(d) in {len(e) + 1, len(e)}, "Invalid diagonal/subdiagonal pair"
	e = np.append([0], e) if len(e) == (len(d) - 1) else e
	d, e = (d.copy(), e.copy())
	Z = np.empty((0, 0), dtype=d.dtype)
	return _eigh_tridiag(d, e, Z, method, maxiter)


# def __tridiag_stemr():
# 	pass
# 	from scipy.linalg import get_lapack_funcs
