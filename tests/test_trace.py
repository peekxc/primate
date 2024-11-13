import numpy as np
from primate.trace import hutch
from primate.operators import MatrixFunction
from primate.stochastic import symmetric, isotropic


def test_hutch():
	rng = np.random.default_rng(1234)
	n = 50
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	est = hutch(A, maxiter=n, seed=rng)
	assert np.abs(A.trace() - est) <= 10 * (1 / np.sqrt(n))

	est, info = hutch(A, maxiter=n, seed=rng, full=True)
	assert isinstance(info.samples, list) and len(info.samples) == n


def test_hutch_mf_identity():
	rng = np.random.default_rng(1234)
	n = 50
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	M = MatrixFunction(A, deg=n, orth=n)

	est1 = hutch(A, maxiter=150, seed=1234)
	est2 = hutch(M, maxiter=150, seed=1234)
	assert np.isclose(est1, est2, atol=1e-6)


def bench_slq():
	from primate.operators import Toeplitz, MatrixFunction

	c = np.random.uniform(size=500)
	T = Toeplitz(c)
	f = lambda x: x + 0.50
	M = MatrixFunction(T, fun=f, deg=20)

	from timeit import timeit

	timeit(lambda: hutch(T, maxiter=150, atol=0.0), number=10)
	timeit(lambda: hutch(M, maxiter=150, atol=0.0), number=10)

	# %%
	G = T @ np.eye(T.shape[0])
	ew, U = np.linalg.eigh(G)
	GM = U @ np.diag(f(ew)) @ U.T

	from primate.trace import hutch

	s1 = hutch(T, fun=f, maxiter=150, atol=0.0, deg=20, seed=0)
	s2 = hutch(M, maxiter=150, atol=0.0, seed=0)
	s3 = hutch(GM, maxiter=150, atol=0.0, seed=0)

	s1
	s2

	v = np.random.choice([-1, +1], size=M.shape[0])
	v = np.random.uniform(low=0, high=1, size=M.shape[0])

	# (v.T @ (T @ v)).item()
	(v.T @ (GM @ v)).item()
	(v.T @ (M @ v)).item()
	M.quad(v).item()

	# np.linalg.norm(v)**2

	# from primate.operators import matrix_function
	# matrix_function(T, fun=lambda x: x + 0.50, v=v, deg=20)

	from primate.stochastic import isotropic

	v = isotropic((M.shape[0], 1))

	np.linalg.norm(v)

	M.quad(v).item()

	(v.T @ (GM @ v)).item()
	M.shape[0] * M.quad(v)
