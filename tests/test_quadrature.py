import numpy as np
from primate.lanczos import lanczos
from primate.random import symmetric
from primate.quadrature import lanczos_quadrature


def test_quadrature():
	rng = np.random.default_rng(seed=1234)
	A = symmetric(50, seed=rng)
	quad_ests = []
	for _ in range(100):
		v = rng.uniform(size=A.shape[1], low=0, high=1)
		v /= np.linalg.norm(v)
		a, b = lanczos(A, deg=A.shape[1], v0=v)
		nodes, weights = lanczos_quadrature(a, b, deg=30, quad="gw")
		quad_ests.append(np.sum(nodes * weights))
	tr_est = np.mean(quad_ests) * A.shape[1]
	assert np.max(np.abs(tr_est - A.trace())) <= 0.10 * A.trace()

	## TODO: fix fttr
	# rng = np.random.default_rng(seed=1234)
	# quad_ests = []
	# for _ in range(1000):
	# 	v = rng.uniform(size=A.shape[1], low=0, high=1)
	# 	v /= np.linalg.norm(v)
	# 	a, b = lanczos(A, deg=A.shape[1], v0=v)
	# 	nodes, weights = lanczos_quadrature(a, b, deg=30, quad="fttr")
	# 	quad_ests.append(np.sum(nodes * weights))
	# tr_est = np.mean(quad_ests) * A.shape[1]
	# assert np.max(np.abs(tr_est - A.trace())) <= 0.10 * A.trace()
