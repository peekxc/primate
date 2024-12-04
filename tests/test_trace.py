import numpy as np
from primate.estimators import EstimatorResult
from primate.trace import hutch, hutchpp, xtrace
from primate.operators import MatrixFunction
from primate.random import symmetric, isotropic


def test_hutch():
	rng = np.random.default_rng(1234)
	n = 54
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	est = hutch(A, seed=rng)
	assert np.abs(A.trace() - est) <= 10 * (1 / np.sqrt(n))

	est, info = hutch(A, seed=rng, full=True)
	assert isinstance(info, EstimatorResult)

	def counter():
		cc = 0

		def _increment(result):
			nonlocal cc
			cc += 1

		return _increment

	counter_cb = counter()
	hutch(A, callback=counter_cb)
	assert counter_cb.__closure__[0].cell_contents > 0

	# assert isinstance(info.criterion.message(info.estimator), str)
	# assert isinstance(info.samples, list) and len(info.samples) == n


def test_hutchpp():
	rng = np.random.default_rng(1234)
	n = 54
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	est = hutchpp(A, m=n, seed=rng)
	assert np.abs(A.trace() - est) <= 1 * (1 / np.sqrt(n))

	est, info = hutchpp(A, m=n, seed=rng, full=True)
	assert isinstance(info, EstimatorResult)


def test_hutch_mf_identity():
	rng = np.random.default_rng(1234)
	n = 50
	ew = rng.uniform(size=n, low=1 / n, high=1.0)
	A = symmetric(n, pd=True, ew=ew, seed=rng)
	M = MatrixFunction(A, deg=n, orth=n)

	est1 = hutch(A, converge="count", count=150, seed=1234)
	est2 = hutch(M, converge="count", count=150, seed=1234)
	assert np.isclose(est1, est2, atol=1e-6)


def test_xtrace():
	## Ensure different batch sizes work with xtrace
	rng = np.random.default_rng(1234)
	A = rng.uniform(size=(50, 50))
	for nb in [1, 2, 3, 5, 10, 20, 50]:
		rng = np.random.default_rng(1234)
		est = xtrace(A, batch=nb, seed=rng, verbose=1, converge="count", count=50)
		err = np.abs(A.trace() - est)
		assert np.isclose(err, 0.0)

	x, info = xtrace(A, full=True, converge="confidence", record=True, atol=0, rtol=0.0)
	assert info.estimator.values != []
	assert isinstance(info, EstimatorResult)
	assert np.abs(A.trace() - x) < 1e-6


def test_xtrace2():
	from primate.trace import xtrace
	from primate.random import symmetric

	rng = np.random.default_rng(1234)  # for reproducibility
	A = symmetric(150, pd=True, seed=rng)  # random PD matrix
	assert np.isclose(xtrace(A), A.trace())

	estimates = []
	xtrace(A, batch=1, full=True, seed=rng, callback=lambda res: estimates.append(res.estimate))
	assert np.std(estimates) < 0.60
