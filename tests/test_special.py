import numpy as np

from primate.random import symmetric
from primate.special import softsign, param_callable


def test_softsign():
	x = np.linspace(-1, 1, 1000)
	norms = []
	for q in range(10):
		norms.append(np.linalg.norm(softsign(x, q=q), ord=1))
	assert np.all(np.diff(norms) >= 0)


def test_mf_quantities():
	rng = np.random.default_rng(1234)
	ew = rng.uniform(size=50, low=-1.0, high=+1.0)
	A = symmetric(len(ew), ew=ew)
	np.linalg.norm(A, "fro")
	np.sum(np.abs(ew))
	np.sqrt(np.sum(ew**2))
	np.sum(ew)
	pass
