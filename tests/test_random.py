import numpy as np
from scipy.stats import normaltest
from primate.random import isotropic, symmetric, haar


def test_isotropic():
	rng = np.random.default_rng(seed=1234)
	for method in ["rademacher", "sphere", "normal"]:
		S = isotropic(size=(1500, 5), pdf=method, seed=rng)
		ES = sum([np.outer(s, s) for s in S]) / len(S)
		assert np.max(np.abs(ES - np.eye(5))) <= 0.10
		if method == "rademacher":
			assert list(np.unique(S.ravel())) == [-1, +1]
		elif method == "sphere":
			assert np.allclose(np.linalg.norm(S, axis=1), np.sqrt(S.shape[1]))
		elif method == "normal":
			assert normaltest(S.ravel()).pvalue >= 0.05  # ensure p-value is large

	rng = np.random.default_rng(seed=1234)
	S1 = isotropic(size=(150, 5), seed=rng)

	rng = np.random.default_rng(seed=1234)
	S2 = isotropic(size=(150, 5), seed=rng)
	assert np.allclose(S1, S2)


def test_haar():
	rng = np.random.default_rng(1234)
	A = haar(25, ew=np.ones(25), seed=rng)
	assert np.allclose(A, np.eye(25))
	A = haar(25, seed=rng)
	assert not np.all(A == A.T)


def test_symmetric():
	rng = np.random.default_rng(1234)
	ew = rng.uniform(size=25)
	A = symmetric(25, ew=ew, seed=rng)
	assert np.allclose(A, A.T)
	assert np.allclose(np.sort(ew), np.sort(np.linalg.eigvalsh(A)))
