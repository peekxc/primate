import numpy as np
from primate.stats import Covariance, confidence_interval


def test_Covariance():
	rng = np.random.default_rng(1234)
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


def test_confidence_interval():
	rng = np.random.default_rng(1234)
	samples = rng.normal(size=1500, loc=0, scale=1 / 2)
	ci_normal = confidence_interval(samples, confidence=0.95, sdist="normal")
	ci_tdist = confidence_interval(samples, confidence=0.95, sdist="t")
	assert np.max(np.abs(np.array(ci_normal) - np.array(ci_tdist))) <= 1e-4
