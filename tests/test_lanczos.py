"""Testing module for lanczos.py"""

import numpy as np
from numpy.random import default_rng
from primate import _lanczos
from primate.lanczos import lanczos
from scipy.linalg import eigvalsh_tridiagonal


def test_lanczos():
	rng = default_rng(seed=1234)
	d = 50
	A = rng.uniform(size=(d, d))
	A @= A.T
	v0 = rng.uniform(size=A.shape[1])
	a, b = lanczos(A, v0=v0, deg=d, orth=d)
	ew_lan = eigvalsh_tridiagonal(a, b)
	ew_dac = np.linalg.eigvalsh(A)
	assert np.allclose(ew_lan, ew_dac), "Eigenvalues not similar"
