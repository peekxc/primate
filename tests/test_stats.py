import numpy as np
from primate.stats import MeanEstimatorCLT, confidence_interval


def test_CLT():
	rng = np.random.default_rng(1234)
	mu = 5.0
	est = MeanEstimatorCLT(confidence=0.95)
	samples = rng.normal(size=150, loc=mu, scale=1 / 2)
	est(samples)
	assert est.n_samples == len(samples)
	ci_test = np.array([est.mu_est - est.margin_of_error, est.mu_est + est.margin_of_error])
	ci_true = np.array(confidence_interval(samples, confidence=0.95, sdist="normal"))
	assert np.allclose(ci_test, ci_true)

	## TODO: test the statistcial difference between many trials
	containing_intervals = 0
	for _ in range(1500):
		est = MeanEstimatorCLT(confidence=0.95)
		intervals = []
		while not est.converged():
			est(rng.normal(size=30, loc=mu, scale=1 / 2))
			intervals.append([est.mu_est - est.margin_of_error, est.mu_est + est.margin_of_error])
		interval = [est.mu_est - est.margin_of_error, est.mu_est + est.margin_of_error]
		containing_intervals += interval[0] <= mu and mu <= interval[1]
	assert abs((containing_intervals / 1500) - 0.95) < (100 / 1500)


def test_confidence_interval():
	rng = np.random.default_rng(1234)
	samples = rng.normal(size=1500, loc=0, scale=1 / 2)
	ci_normal = confidence_interval(samples, confidence=0.95, sdist="normal")
	ci_tdist = confidence_interval(samples, confidence=0.95, sdist="t")
	assert np.max(np.abs(np.array(ci_normal) - np.array(ci_tdist))) <= 1e-4
