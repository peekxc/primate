"""Testing module for tridiagonal.py"""

import numpy as np
from numpy.random import default_rng
from primate2.lanczos import lanczos
from primate2.tridiag import eigvalsh_tridiag, eigh_tridiag
from primate2.stochastic import symmetric
from primate2.tqli import tqli


def test_tqli():
	rng = default_rng(seed=1234)
	d = 50
	ew = np.sort(rng.uniform(size=d, low=1 / d, high=1))
	A = symmetric(d, seed=rng, pd=True, ew=ew)
	v = rng.uniform(size=d)
	a, b = lanczos(A, v0=v, deg=d, orth=d)
	d, e = a.copy(), np.append([0], b)
	Z = np.empty((0, 0), dtype=A.dtype)
	tqli(d, e, Z, 30)
	assert np.allclose(np.sort(d), ew)
	assert np.allclose(e, 0.0)
	np.max(np.abs(np.sort(d) - ew))


def test_tridiag():
	for seed in [1234, 4756, 43, 102]:
		rng = default_rng(seed=seed)
		d = 150
		ew = np.sort(rng.uniform(size=d, low=1 / d, high=1))
		A = symmetric(d, seed=rng, pd=True, ew=ew)
		v = rng.uniform(size=d)
		a, b = lanczos(A, v0=v, deg=d, orth=d)
		for method in ["tqli", "mrrr"]:
			ew_test = np.sort(eigvalsh_tridiag(a, b, method=method))
			assert np.allclose(ew_test, ew), f"Eigenvalue test failed for method = {method}"
			assert np.max(np.abs(ew_test - ew)) <= 1e-14

		for method in ["tqli", "mrrr"]:
			ew_test, ev_test = eigh_tridiag(a, b, method=method)
			B = ev_test.T @ ev_test
			assert np.allclose(B - np.diag(B.diagonal()), 0.0)
			assert np.allclose(B.diagonal(), 1.0)
			assert np.allclose(np.sort(ew_test), ew)
