import numpy as np
from primate.lanczos import lanczos
from primate.random import symmetric
from primate.quadrature import lanczos_quadrature


def test_quadrature():
	rng = np.random.default_rng(seed=1234)
	A = symmetric(50, seed=rng, pd=True)
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
	rng = np.random.default_rng(seed=1234)
	quad_ests = []
	for _ in range(1000):
		v = rng.uniform(size=A.shape[1], low=0, high=1)
		v /= np.linalg.norm(v)
		a, b = lanczos(A, deg=A.shape[1], v0=v)
		nodes, weights = lanczos_quadrature(a, b, deg=30, quad="fttr")

		quad_ests.append(np.sum(nodes * weights))
	tr_est = np.mean(quad_ests) * A.shape[1]
	# assert np.max(np.abs(tr_est - A.trace())) <= 0.10 * A.trace()


def test_fttr_basic():
	from scipy.sparse import spdiags

	alpha = np.array([1, 1, 1])
	beta = np.array([1, 1, 0])
	T = spdiags(data=[beta, alpha, np.roll(beta, 1)], diags=(-1, 0, +1), m=3, n=3).todense()
	ew, ev = np.linalg.eigh(T)

	a = alpha
	b = np.append(0, beta[:-1])
	mu_0 = np.sum(np.abs(ew))
	p0 = lambda x: 1 / np.sqrt(mu_0)
	p1 = lambda x: (x - a[0]) * p0(x) / b[1]
	p2 = lambda x: ((x - a[1]) * p1(x) - b[1] * p0(x)) / b[2]
	p = lambda x: np.array([p0(x), p1(x), p2(x)])
	weights_fttr = np.reciprocal([np.sum(p(lam) ** 2) for lam in ew])

	## Forward three-term recurrence relation (fttr)
	assert np.allclose(weights_fttr, mu_0 * np.ravel(ev[0, :]) ** 2)


def test_fftr():
	from scipy.linalg import toeplitz
	from scipy.sparse import spdiags

	rng = np.random.default_rng(1234)
	n = 8
	A = toeplitz(np.arange(n)).astype(np.float64)  # symmetric(n)
	v0 = rng.uniform(size=A.shape[1])
	alpha, beta = lanczos(A, v0=v0, deg=n, orth=n - 1)
	a, b = alpha, np.append([0], beta)
	T = spdiags(data=[np.roll(b, -1), a, b], diags=(-1, 0, +1), m=n, n=n).todense()
	ew, ev = np.linalg.eigh(T)

	a, b = np.diag(T, 0).copy(), np.append([0], np.diag(T, 1)).copy()
	mu_0 = np.sum(np.abs(ew))
	fttr_nodes, fttr_weights_run = lanczos_quadrature(a, b, deg=30, quad="fttr")
	fttr_weights_true = np.ravel(ev[0, :]) ** 2
	assert np.allclose(fttr_weights_run, fttr_weights_true)

	quad_test = np.sum(fttr_nodes * fttr_weights_run)
	quad_true = np.sum(fttr_weights_true * ew)
	assert np.isclose(quad_test, quad_true, atol=1e-10)
