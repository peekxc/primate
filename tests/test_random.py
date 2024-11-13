import numpy as np
from scipy.stats import normaltest
from primate.stochastic import isotropic


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
