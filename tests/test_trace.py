import numpy as np
from primate2.estimators import hutch
from primate2.stochastic import symmetric


def test_hutch():
	rng = np.random.default_rng(1234)
	n = 50
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	samples = hutch(A, seed=rng)
	assert np.abs(A.trace() - np.mean(samples)) <= 10 * (1 / np.sqrt(n))


def test_hutch_mf():
	rng = np.random.default_rng(1234)
	n = 50
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	t = np.array([-1e-3, -1e-2, -1e-1])
	fun = lambda x: np.exp(-t * x[:, np.newaxis])

	hutch(A, fun=fun, maxiter=10, stop=None)

	# from scipy.stats import trim_mean
	# hutch(A, reduce=)
	# A.trace()

	pass
