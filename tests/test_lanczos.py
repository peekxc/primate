"""Testing module for lanczos.py"""

import numpy as np
from numpy.random import default_rng
from scipy.linalg import eigvalsh_tridiagonal

from primate import _lanczos
from primate.lanczos import lanczos, rayleigh_ritz
from primate.random import symmetric


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


def test_rayleigh():
	rng = default_rng(seed=1234)
	d = 50
	ew = rng.uniform(size=d, low=0, high=1)
	A = symmetric(d, ew=ew, seed=rng)
	v0 = rng.uniform(size=A.shape[1])
	rw = rayleigh_ritz(A, 20, v0=v0)
	assert np.isclose(np.max(rw), np.max(ew), atol=1e-2)
	assert np.isclose(np.min(rw), np.min(ew), atol=1e-2)
