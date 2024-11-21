import numpy as np


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

	from primate.random import isotropic

	v = isotropic((M.shape[0], 1))

	np.linalg.norm(v)

	M.quad(v).item()

	(v.T @ (GM @ v)).item()
	M.shape[0] * M.quad(v)
