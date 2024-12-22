import numpy as np
from scipy.stats import normaltest
from primate.random import isotropic, symmetric, haar


def test_isotropic():
	rng = np.random.default_rng(seed=1235)
	for method in ["rademacher", "sphere", "normal"]:
		S = isotropic(size=(5, 1500), pdf=method, seed=rng)
		ES = sum([np.outer(s, s) for s in S.T]) / S.shape[1]
		assert np.max(np.abs(ES - np.eye(S.shape[0]))) <= 0.15
		if method == "rademacher":
			assert list(np.unique(S.ravel())) == [-1, +1]
		elif method == "sphere":
			assert np.allclose(np.linalg.norm(S, axis=0), np.sqrt(S.shape[0]))
		elif method == "normal":
			assert normaltest(S.ravel()).pvalue >= 0.05  # ensure p-value is large
	S1 = isotropic(size=(150, 5), seed=1234)
	S2 = isotropic(size=(150, 5), seed=1234)
	assert np.allclose(S1, S2)


def test_iso_order():
	rng = np.random.default_rng(seed=1234)
	ew = rng.uniform(size=20, low=0.0, high=1.0)
	A = symmetric(20, ew=ew)

	rng = np.random.default_rng(seed=1234)
	qe = []
	for i in range(150):
		v = np.ravel(isotropic(20, pdf="sphere", seed=rng))
		qe.append(v @ A @ v)
	qe = np.ravel(qe)

	rng = np.random.default_rng(seed=1234)
	V = isotropic((A.shape[1], 150), pdf="sphere", seed=rng)
	qe2 = np.diag(V.T @ A @ V)

	assert np.allclose(np.ravel(qe), np.ravel(qe2))


def test_Isotropic():
	from primate.random import Isotropic, isotropic
	import timeit

	Iso = Isotropic(size=(50, 1500), pdf="normal", seed=1234)
	t1 = timeit.timeit(lambda: Iso.fill(), number=500)
	t2 = timeit.timeit(lambda: isotropic((50, 1500), pdf="normal"), number=500)
	assert t1 <= 1.05 * t2

	Iso = Isotropic(size=(50, 1500), pdf="normal", seed=1234)
	t1 = timeit.timeit(lambda: Iso.fill(), number=500)
	t2 = timeit.timeit(lambda: isotropic(Iso.values.shape, pdf="normal"), number=500)
	assert t1 <= 1.05 * t2

	Iso = Isotropic(size=(50, 5), pdf="sphere", seed=1234, threads=1)
	Iso.fill()
	Y2 = isotropic((50, 5), "sphere", seed=1234)
	assert np.allclose(Iso.values, Y2)

	# from line_profiler import LineProfiler
	# profile = LineProfiler()
	# profile.add_function(Iso.fill)
	# profile.add_function(isotropic_inplace)
	# profile.add_function(isotropic)
	# profile.enable_by_count()
	# for _ in range(500):
	# 	Iso.fill()
	# for _ in range(500):
	# 	y = isotropic((1500, 50), pdf="rademacher")
	# profile.print_stats()

	# Iso.values


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
