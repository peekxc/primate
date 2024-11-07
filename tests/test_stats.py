import numpy as np
from primate2.stats import MeanEstimatorCLT, confidence_interval


def test_CLT():
	rng = np.random.default_rng(1234)
	mu = 5.0
	est = MeanEstimatorCLT()
	samples = rng.normal(size=150, loc=mu, scale=1 / 2)
	assert est.n_samples == len(samples)
	ci_test = np.array([est.mu_est - est.margin_of_error, est.mu_est + est.margin_of_error])
	ci_true = np.array(confidence_interval(samples, confidence=0.95, sdist="normal"))
	assert np.allclose(ci_test, ci_true)

	## TODO: test the statistcial difference between many trials
	# intervals = []
	# for _ in range(1500):
	# est = MeanEstimatorCLT()
	# while not est.converged():
	# 	est(rng.normal(size=10, loc=mu, scale=1 / 2))
	# 	intervals.append([est.mu_est - est.margin_of_error, est.mu_est + est.margin_of_error])
	# intervals = np.array(intervals)
	# containing_intervals = np.sum(np.logical_and(intervals[:, 0] <= mu, mu <= intervals[:, 1]))
	# containing_intervals / 1500

	# valid_ci = 0
	# for _ in range(1500):
	# 	samples = rng.normal(size=150, loc=mu, scale=1 / 2)
	# 	lb, ub = sample_mean_cinterval(samples, confidence=0.95, sdist="normal")
	# 	valid_ci += lb <= mu and mu <= ub

	# assert (est.mu_est - est.margin_of_error) <= 5 <= (est.mu_est + est.margin_of_error)
	# from bokeh.plotting import show

	# show(est.plot(samples))


def test_confidence_interval():
	rng = np.random.default_rng(1234)
	samples = rng.normal(size=1500, loc=0, scale=1 / 2)
	ci_normal = confidence_interval(samples, confidence=0.95, sdist="normal")
	ci_tdist = confidence_interval(samples, confidence=0.95, sdist="t")
	assert np.max(np.abs(np.array(ci_normal) - np.array(ci_tdist))) <= 1e-4
