from typing import Callable

import numpy as np
from primate.estimators import (
	ConfidenceCriterion,
	ControlVariableEstimator,
	ConvergenceCriterion,
	CountCriterion,
	KneeCriterion,
	MeanEstimator,
	ToleranceCriterion,
	convergence_criterion,
)
from primate.stats import confidence_interval


def test_MeanEstimator():
	rng = np.random.default_rng(1234)
	mu = MeanEstimator()
	samples = []
	for _ in range(25):
		samples.extend(rng.normal(size=10))
		mu.update(samples[-10:])
	assert np.allclose(np.mean(samples), mu.mean)
	assert isinstance(mu.estimate, float)


## From Variance reduction book
def test_ControlVariableEstimator():
	rng = np.random.default_rng(1235)
	a = np.array([1, 2, 3, 1, 2])
	h = lambda x: np.min([x[0] + x[3], x[0] + x[2] + x[4], x[1] + x[2] + x[3], x[1] + x[4]])

	## Ground truth
	mu = 1339 / 1440

	## Control variable
	mu_cv = 15 / 16
	h_cv = lambda x: np.min([x[0] + x[3], x[1] + x[4]])

	## Compare estimators
	n_efficient = 0
	for _ in range(150):
		U = rng.uniform(low=0, high=1, size=(250, 5))
		y = np.apply_along_axis(h, 1, U * a)
		y_cv = np.apply_along_axis(h_cv, 1, U * a)
		est1, est2 = MeanEstimator(), ControlVariableEstimator(mu_cv)
		est1.update(y)
		est2.update(y, y_cv)
		n_efficient += np.linalg.norm(est2.estimate - mu) <= np.linalg.norm(est1.estimate - mu)
	assert n_efficient >= 120


def test_CountCriterion():
	rng = np.random.default_rng(1234)
	mu = MeanEstimator()
	cc = CountCriterion(10)
	assert not cc(mu)
	for _ in range(9):
		mu.update(rng.uniform(size=1, low=-1, high=+1).item())
		assert not cc(mu)
	mu.update(rng.uniform(size=1, low=-1, high=+1).item())
	assert len(mu) == 10
	assert cc(mu)


def test_ToleranceCriterion():
	rng = np.random.default_rng(1234)
	mu = MeanEstimator(15)
	cc = ToleranceCriterion(atol=0, rtol=0.10, ord=1)
	while not cc(mu):
		mu.update(rng.uniform(size=(1, 15), low=-1, high=+1))
	error = np.linalg.norm(mu.delta, ord=1)
	assert error < (np.linalg.norm(mu.estimate, ord=1) * 0.10)
	assert (np.linalg.norm(mu.estimate, ord=1) * 0.10) <= 0.20


def test_ConfidenceCriterion():
	## TODO: better statistical difference test between many trials
	rng = np.random.default_rng(1234)
	mu = 5.0
	containing_intervals = 0
	for _ in range(1500):
		atol = 0.50
		est = MeanEstimator()
		cc = ConfidenceCriterion(confidence=0.95, atol=atol, rtol=0.0)
		while not cc(est):
			est.update(rng.normal(size=5, loc=mu, scale=1 / 2))
		containing_intervals += np.abs(mu - est.estimate) <= atol
	assert abs((containing_intervals / 1500) - cc.confidence) < (100 / 1500)


def test_KneeCriterion():
	rng = np.random.default_rng(1234)
	mu = MeanEstimator(record=True)
	cc = KneeCriterion(S=1.0)
	assert not cc(mu)
	while not cc(mu):
		mu.update(rng.normal(size=1, loc=0, scale=1))
	assert cc(mu)
	assert np.abs(mu.delta) <= 0.15


def test_CriterionComposability():
	rng = np.random.default_rng(1234)
	mu = MeanEstimator()
	cc1 = CountCriterion(200)
	cc2 = ConfidenceCriterion(confidence=0.95, atol=0.50, rtol=0.0)

	## Test AND
	assert not cc1(mu) and not cc2(mu)
	cc = cc1 & cc2
	assert isinstance(cc, Callable) and cc(mu) is False
	while not (cc1(mu) and cc2(mu)):
		assert not cc(mu)
		mu.update(rng.uniform(size=1, low=-1, high=+1).item())
	assert cc1(mu) and cc2(mu) and cc(mu)

	## Test OR
	mu = MeanEstimator()
	cc = cc1 | cc2
	assert isinstance(cc, Callable) and cc(mu) is False
	while not (cc1(mu) or cc2(mu)):
		assert not cc(mu)
		mu.update(rng.uniform(size=1, low=-1, high=+1).item())
	assert (cc1(mu) or cc2(mu)) and cc(mu)


def test_converge_param():
	cc = convergence_criterion("count", count=10)
	assert isinstance(cc, ConvergenceCriterion)
	cc = convergence_criterion("tolerance", atol=0.0, rtol=0.0)
	assert isinstance(cc, ConvergenceCriterion)
	cc = convergence_criterion("confidence", confidence=0.95)
	assert isinstance(cc, ConvergenceCriterion)
	cc = convergence_criterion("knee", S=10.0)
	assert isinstance(cc, ConvergenceCriterion)
