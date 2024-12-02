import numpy as np
from primate.diagonal import diag, xdiag


def test_diag():
	rng = np.random.default_rng(1234)
	A = rng.normal(size=(50, 50))
	d, info = diag(A, converge="tolerance", atol=0.10, rtol=0.0, full=True)
	assert info.criterion(info.estimator)
	assert np.linalg.norm(info.estimator.delta, 2) <= 0.10
	d = diag(A, converge="tolerance", atol=0.0, rtol=0.001)
	assert np.linalg.norm(A.diagonal() - d, 2) < 10.0


def test_xdiag():
	rng = np.random.default_rng(1234)
	A = rng.normal(size=(150, 150))

	## Ensure the length is right
	d = xdiag(A, m=10)
	assert isinstance(d, np.ndarray) and len(d) == A.shape[0]

	## Ensure error is decreasing
	errors = []
	budget = np.linspace(2, 2 * A.shape[0], 10).astype(int)
	for m in budget:
		d = xdiag(A, m, pdf="signs", seed=rng)
		errors.append(np.linalg.norm(np.diag(A) - d))
		# print(f"Error: {np.linalg.norm(np.diag(A) - d)}")

	y = np.array(errors)
	B = np.c_[budget, np.ones(len(budget))]
	m, c = np.linalg.lstsq(B, y)[0]
	assert m < -0.10, "Error is not decreasing appreciably"


# def test_diagonal():
# 	rng = np.random.default_rng(1234)
# 	A = rng.normal(size=(50, 50))
# 	d, info = diag(A, full=True)

# 	from primate.plotting import figure_sequence

# 	show(figure_sequence(info.samples))
# 	np.linalg.norm(d - A.diagonal())

# 	p = figure(width=250, height=250)
# 	p.line(np.arange(len(info.samples)) / 48, info.samples)
# 	p.scatter(np.arange(len(info.samples)) / 48, info.samples)
# 	show(p)

# 	x = np.arange(len(info.samples)).astype(np.float64)
# 	y = info.samples
# 	x /= np.max(x)
# 	y /= np.max(y)

# 	from kneed import KneeLocator

# 	kneedle = KneeLocator(x, y, S=10.0, curve="convex", direction="decreasing", online=True)

# 	y = np.abs(1 - y)
# 	# 10.0 * (np.cumsum(np.diff(x))/np.arange(1, len(x)))

# 	from typing import Union

# 	x, y = x[:4], y[:4]
# 	t_thresh = np.max(y - x) - 1.0 * np.mean(np.diff(x))
# 	max_diff = np.max(y - x)

# 	p = figure(width=250, height=250)
# 	p.line(x, y)
# 	p.scatter(x, y)
# 	# p.vspan(x=kneedle.knee, color="red")
# 	p.line(x, y - x, color="blue")
# 	p.hspan(y=t_thresh, color="green", line_dash="dashed")
# 	show(p)

# 	est = KneedleErrorEstimator(S=10.0)
# 	for yi in y:
# 		est.update(yi)
# 		print(est.converged())
