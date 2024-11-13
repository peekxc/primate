import numpy as np
from primate.stats import CentralLimitEstimator, Covariance, confidence_interval, MeanEstimator


def test_MeanEstimator():
	rng = np.random.default_rng()
	mu = MeanEstimator()
	samples = []
	for _ in range(25):
		samples.extend(rng.normal(size=10))
		mu.update(samples[-10:])
		assert not mu.converged()
	assert np.allclose(np.mean(samples), mu.mean)
	assert isinstance(mu.estimate, float)


def test_Covariance():
	rng = np.random.default_rng()
	C = Covariance(dim=1)
	samples = []
	for _ in range(25):
		samples.extend(rng.normal(size=10))
		C.update(samples[-10:])
		assert np.isclose(np.var(samples, ddof=1), C.covariance())
		assert np.isclose(np.mean(samples), C.mean)
		assert len(samples) == C.n

	C = Covariance(dim=2)
	samples.clear()
	for _ in range(25):
		samples.extend(rng.normal(size=(10, 2)))
		C.update(samples[-10:])
		assert np.allclose(np.cov(samples, rowvar=False, ddof=1), C.covariance())
		assert np.allclose(np.mean(samples, axis=0), C.mean)
		assert len(samples) == C.n


def test_CLT():
	rng = np.random.default_rng(1234)
	mu = 5.0
	sc = CentralLimitEstimator(confidence=0.95)
	samples = rng.normal(size=150, loc=mu, scale=1 / 2)
	sc.update(samples)
	assert sc.n_samples == len(samples)
	ci_test = np.array([sc.cov.mean - sc.margin_of_error, sc.cov.mean + sc.margin_of_error])
	ci_true = np.array(confidence_interval(samples, confidence=0.95, sdist="normal"))
	assert np.allclose(ci_test, ci_true)

	## TODO: test the statistcial difference between many trials
	containing_intervals = 0
	for _ in range(1500):
		sc = CentralLimitEstimator(confidence=0.95)
		intervals = []
		while not sc.converged():
			sc.update(rng.normal(size=30, loc=mu, scale=1 / 2))
			intervals.append([sc.cov.mean - sc.margin_of_error, sc.cov.mean + sc.margin_of_error])
		interval = [sc.cov.mean - sc.margin_of_error, sc.cov.mean + sc.margin_of_error]
		containing_intervals += interval[0] <= mu and mu <= interval[1]
	assert abs((containing_intervals / 1500) - 0.95) < (100 / 1500)


def test_confidence_interval():
	rng = np.random.default_rng(1234)
	samples = rng.normal(size=1500, loc=0, scale=1 / 2)
	ci_normal = confidence_interval(samples, confidence=0.95, sdist="normal")
	ci_tdist = confidence_interval(samples, confidence=0.95, sdist="t")
	assert np.max(np.abs(np.array(ci_normal) - np.array(ci_tdist))) <= 1e-4
