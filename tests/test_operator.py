import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from primate2.operators import matrix_function, MatrixFunction
from primate2.stochastic import symmetric
from primate2.lanczos import lanczos
from primate2.tridiag import eigh_tridiag
from primate2.special import param_callable, _builtin_matrix_functions


def test_operator_logic():
	rng = np.random.default_rng(1234)
	n = 100
	A = symmetric(n)
	v = rng.uniform(size=A.shape[1], low=-1, high=1)

	(a, b), Q = lanczos(A, v0=v, deg=n, return_basis=True)  # O(nd)  space
	rw, Y = eigh_tridiag(a, b)  # O(d^2) space
	e1 = np.zeros(len(rw))
	e1[0] = 1
	z = np.linalg.norm(v) * Q @ (Y @ np.diag(rw) @ Y.T @ e1)
	assert np.isclose(np.linalg.norm(z - A @ v), 0.0)
	y = np.linalg.norm(v) * Q @ Y @ (rw * Y[0, :])[:, np.newaxis]
	assert np.allclose(z, y.ravel())

	(a, b), Q = lanczos(A, v0=v, deg=5, return_basis=True)  # O(nd)  space
	rw, Y = eigh_tridiag(a, b)  # O(d^2) space
	e1 = np.zeros(len(rw))
	e1[0] = 1
	z = np.linalg.norm(v) * Q @ (Y @ np.diag(rw) @ Y.T @ e1)
	assert np.isclose(np.linalg.norm(z - A @ v), 0.0)
	y = np.linalg.norm(v) * Q @ Y @ (rw * Y[0, :])[:, np.newaxis]
	assert np.allclose(z, y.ravel())


def test_operator_interface():
	rng = np.random.default_rng(1234)
	n = 100
	A = symmetric(n)
	for deg in [n, n - 5, n - 10, n - 30, n - 50, n - 75]:
		M = MatrixFunction(A, deg=A.shape[0], orth=n, dtype=np.float64)
		for _ in range(3):
			v = rng.uniform(size=A.shape[1], low=-1, high=1)
			v_scale = np.linalg.norm(v)
			(a, b), Q = lanczos(A, v0=v, deg=deg, return_basis=True)  # O(nd)  space
			rw, Y = eigh_tridiag(a, b)  # O(d^2) space
			z1 = M._matvec(v.copy()).ravel()
			z2 = (v_scale * Q @ Y @ (rw * Y[0, :])[:, np.newaxis]).ravel()
			assert np.allclose(z1, z2)
	M = MatrixFunction(A, deg=A.shape[0], orth=n, dtype=np.float64)
	assert isinstance(M, LinearOperator), "Matrix function is not a linear operator"
	assert np.allclose(A @ v, M @ v)

	L = aslinearoperator(A)
	M = MatrixFunction(L, deg=A.shape[0], orth=n, dtype=np.float64)
	assert np.allclose(A @ v, M @ v)


def test_spectral():
	rng = np.random.default_rng(1234)
	n = 100
	A = symmetric(n)
	v = rng.uniform(size=A.shape[1], low=-1, high=1)
	for fun in _builtin_matrix_functions:
		f = param_callable(fun)
		M = MatrixFunction(A, fun=f, deg=A.shape[0])
		ew, ev = np.linalg.eigh(A)
		y = ev @ np.diag(f(ew)) @ ev.T @ v
		z = M @ v
		assert np.allclose(y, z)
