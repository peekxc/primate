import numpy as np

from primate.special import softsign


def test_softsign():
	x = np.linspace(-1, 1, 1000)
	norms = []
	for q in range(10):
		norms.append(np.linalg.norm(softsign(x, q=q), ord=1))
	assert np.all(np.diff(norms) >= 0)
