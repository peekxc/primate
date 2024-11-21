import numpy as np
from primate.linalg import update_trinv


def test_update_inv():
	## 1x1
	B = np.array([[1.0]])
	b = np.array([2.0, 1.0])
	B_new = np.c_[np.vstack((B, 0.0)), b]
	B_ast = update_trinv(np.linalg.inv(B), b)
	assert np.allclose(np.linalg.inv(B_new), B_ast, atol=1e-8)

	## 2x2
	B = np.array([[1.0, 2.0], [0.0, 4.0]])
	b = np.array([5.0, 6.0, 7.0])
	B_new = np.c_[np.vstack((B, np.zeros(B.shape[1]))), b]
	B_ast = update_trinv(np.linalg.inv(B), b)
	assert np.allclose(np.linalg.inv(B_new), B_ast, atol=1e-8)

	## nxn
	rng = np.random.default_rng(1234)
	for n in range(3, 20):
		B = np.triu(rng.uniform(size=(n, n)))
		b = rng.uniform(size=n + 1)
		B_new = np.c_[np.vstack((B, np.zeros(B.shape[1]))), b]
		B_ast = update_trinv(np.linalg.inv(B), b)
		assert np.allclose(np.linalg.inv(B_new), B_ast, atol=1e-8)


def test_qr_insert():
	from scipy.linalg import qr_insert

	rng = np.random.default_rng(1234)
	Y = rng.uniform(size=(5, 2))
	Q, R = np.linalg.qr(Y)
	y = rng.uniform(size=(Y.shape[0], 1))
	Q_hat, R_hat = qr_insert(Q, R, u=y, k=-1, which="col", rcond=None, overwrite_qru=False, check_finite=True)

	# QR update on row or column insertions
