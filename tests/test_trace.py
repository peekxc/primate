import numpy as np 
from numbers import Number
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric
from typing import * 

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

## NOTE: trace estimation only works with isotropic vectors 
def test_girard_fixed():
	from sanity import girard_hutch
	np.random.seed(1234)
	n = 30
	ew = 0.2 + 1.5*np.linspace(0, 5, n)
	Q,R = np.linalg.qr(np.random.uniform(size=(n,n)))
	A = Q @ np.diag(ew) @ Q.T
	A = (A + A.T) / 2
	ew_true = np.linalg.eigvalsh(A)
	tr_est = girard_hutch(A, lambda x: x, nv = n, estimates=False)
	threshold = 0.05*(np.max(ew)*n - np.min(ew)*n)
	assert np.allclose(ew_true, ew)
	assert np.isclose(A.trace() - tr_est, 0.0, atol=threshold)

def test_trace_import():
	import primate.trace
	assert '_trace' in dir(primate.trace)
	from primate.trace import hutch, _trace
	assert 'hutch' in dir(_trace)
	assert isinstance(hutch, Callable)

def test_trace_basic():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 10 
	A = symmetric(n)
	tr_test1 = hutch(A, maxiter=100, seed=5, num_threads=1)
	tr_test2 = hutch(A, maxiter=100, seed=5, num_threads=1)
	tr_true = A.trace()
	assert tr_test1 == tr_test2, "Builds not reproducible!"
	assert np.isclose(tr_test1, tr_true, atol=tr_true*0.05)

def test_trace_identity():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 10 
	A = np.eye(n)
	from primate.operator import matrix_function
	M = matrix_function(A, fun=lambda x: x)
	assert np.isclose(hutch(M), 10)

	hutch(A, fun="identity", info=True)

def test_trace_pdfs():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 50
	A = symmetric(n)
	tr_test1 = hutch(A, maxiter=200, seed=5, num_threads=1, pdf="rademacher")
	tr_test2 = hutch(A, maxiter=200, seed=5, num_threads=1, pdf="normal")
	tr_true = A.trace()
	assert np.isclose(tr_test1, tr_test2, atol=tr_true*0.05)

def test_trace_inputs():
	from primate.trace import hutch
	n = 10 
	A = symmetric(n)
	tr_1 = hutch(A, maxiter=100)
	tr_2 = hutch(csc_array(A), maxiter=100)
	tr_3 = hutch(aslinearoperator(A), maxiter=100)
	assert all([isinstance(t, Number) for t in [tr_1, tr_2, tr_3]]) 

def test_hutch_info():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 25
	A = csc_array(symmetric(n), dtype=np.float32)
	tr_est, info = hutch(A, maxiter = 200, info=True)
	assert isinstance(info, dict) and isinstance(tr_est, Number)
	assert len(info['samples']) == 200
	assert np.all(~np.isclose(info['samples'], 0.0))
	assert np.isclose(tr_est, A.trace(), atol=1.0)

def test_hutch_multithread():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 25
	A = csc_array(symmetric(n), dtype=np.float32)
	tr_est, info = hutch(A, maxiter = 200, atol=0.0, info = True, num_threads=6)
	assert len(info['samples'] == 200)
	assert np.all(~np.isclose(info['samples'], 0.0))
	assert np.isclose(tr_est, A.trace(), atol=1.0)

def test_hutch_clt_atol():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 30
	A = csc_array(symmetric(n), dtype=np.float32)
	
	from primate.stats import sample_mean_cinterval
	tr_est, info = hutch(A, maxiter = 100, num_threads=1, seed=5, info=True)
	tr_samples = info['samples']
	ci = np.array([sample_mean_cinterval(tr_samples[:i], sdist='normal') if i > 1 else [-np.inf, np.inf] for i in range(len(tr_samples))])
	
	## Detect when, for the fixed set of samples, the trace estimator should converge by CLT 
	atol_threshold = (A.trace() * 0.05)
	clt_converged = np.ravel(0.5*np.diff(ci, axis=1)) <= atol_threshold
	assert np.any(clt_converged), "Did not converge!"
	converged_ind = np.flatnonzero(clt_converged)[0]

	## Re-run with same seed and ensure the index matches
	tr_est, info = hutch(A, maxiter = 100, num_threads=1, atol=atol_threshold, seed=5, info=True)
	tr_samples = info['samples']
	converged_online = np.take(np.flatnonzero(tr_samples == 0.0), 0)
	assert converged_online == converged_ind, "hutch not converging at correct index!"

def test_hutch_change():
	from primate.trace import hutch
	np.random.seed(1234)
	n = 30
	A = csc_array(symmetric(n), dtype=np.float32)
	tr_est, info = hutch(A, maxiter = 100, num_threads=1, seed=5, info=True)
	tr_samples = info['samples']
	estimator = np.cumsum(tr_samples) / np.arange(1, 101)
	conv_ind_true = np.flatnonzero(np.abs(np.diff(estimator)) <= 0.001)[0] + 1

	## Test the convergence checking for the atol change method
	tr_est, info = hutch(A, maxiter = 100, num_threads=1, seed=5, info=True, atol=0.001, stop="change")
	conv_ind_test = np.take(np.flatnonzero(info['samples'] == 0), 0)
	assert abs(conv_ind_true - conv_ind_test) <= 1

def test_trace_mf():
	from primate.trace import hutch
	n = 10 
	A = symmetric(n)
	tr_est = hutch(A, fun="identity", maxiter=100, num_threads=1, seed = 5)
	tr_true = A.trace()
	assert np.isclose(tr_est, tr_true, atol=tr_true*0.05)
	tr_est = hutch(A, fun=lambda x: x, maxiter=100, num_threads=1, seed = 5)
	assert np.isclose(tr_est, tr_true, atol=tr_true*0.05)

def test_trace_fftr():
	from primate.trace import hutch
	n = 50 
	A = symmetric(n)

	## Test the fttr against the golub_welsch
	tr_est1, info1 = hutch(A, fun="identity", maxiter=100, seed=5, num_threads=1, info=True, quad="golub_welsch")
	tr_est2, info2 = hutch(A, fun="identity", maxiter=100, seed=5, num_threads=1, info=True, quad="fttr")
	assert np.isclose(tr_est1, tr_est2, atol=tr_est1*0.01)

	## Test accuracy
	assert np.isclose(A.trace(), tr_est1, atol=tr_est1*0.025)

	# from primate.diagonalize import lanczos, _lanczos
	# v0 = np.array([-1, 1, 1,-1, 1,-1, 1,-1,-1, 1]) / np.sqrt(10)
	# a, b = lanczos(A, v0=v0, deg=10)
	# a, b = a, np.append([0], b)
	# _lanczos.quadrature(a, b, 10, 0)


	# from primate.operator import matrix_function
	# M = matrix_function(A, fun="identity")
	# M.method = "fttr"
	# M.quad(np.random.choice([-1.0, +1.0], size=n))

	## TODO 
	# tr_est, info = hutch(A, fun=lambda x: x, maxiter=100, seed=5, num_threads=1, info=True)
	# assert np.isclose(tr_est, tr_true, atol=tr_true*0.05)
	# for s in range(15000):
	#   est, info = hutch(A, fun="identity", deg=2, maxiter=200, num_threads=1, seed=591, info=True)
	#   assert not np.isnan(est)
	# for s in range(15000):
	#   est, info = hutch(A, fun="identity", deg=20, maxiter=200, num_threads=8, seed=-1, info=True)
	#   assert not np.isnan(est)

	# from primate.operator import matrix_function
	# M = matrix_function(A, fun="identity", deg=20)
	# for s in range(15000):
	#   v0 = np.random.choice([-1, 1], size=M.shape[0])
	#   assert not np.isnan(M.quad(v0))


	# from primate.diagonalize import lanczos
	# lanczos(A, v0=v0, rtol=M.rtol, deg=M.deg, orth=M.orth)
		# if np.any(np.isnan(info['samples']))

def test_quad_sum():
	from primate.trace import _trace
	np.random.seed(1234)
	n = 100
	A = symmetric(n)
	Q = np.linalg.qr(A)[0]
	test_quads = _trace.quad_sum(A, Q)
	true_quads = (Q.T @ (A @ Q)).diagonal()
	assert np.allclose(test_quads, true_quads)
	assert np.isclose(np.sum(test_quads), np.sum(true_quads))
	test_quads = _trace.quad_sum(A, Q[:,:10])
	true_quads = (Q[:,:10].T @ (A @ Q[:,:10])).diagonal()
	assert np.allclose(test_quads, true_quads)
	assert np.isclose(np.sum(test_quads), np.sum(true_quads))

def test_hutch_pp():
	from primate.trace import hutchpp
	np.random.seed(1234)
	n = 100
	A = symmetric(n)
	test_trace = hutchpp(A, mode="reduced", seed=1)
	true_trace = A.trace()
	assert np.isclose(test_trace, true_trace, atol=1.0)
	test_trace = hutchpp(A, mode="full", seed=1)
	assert np.isclose(test_trace, true_trace, atol=1.0)

	# import timeit
	# timeit.timeit(lambda: hutchpp(A, mode="full"), number=1000)
	# timeit.timeit(lambda: hutchpp(A, mode="reduced"), number=1000)

	# ## Raw isotropic random vectors 
	# m = 20 * 3
	# N: int = A.shape[0]
	# M: int = 2 * m // 3
	# f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	# W = np.random.choice([-1.0, +1.0], size=(N, M)).astype(f_dtype)
	# W1, W2 = W[:,:(m // 3)], W[:,(m // 3):]
	# Q = np.linalg.qr(A @ W2)[0]

	# ## Start with estimate using the largest eigen-spaces
	# bulk_tr = (Q.T @ (A @ Q)).trace()

	# ## Estimate residual via Girard
	# # from scipy.sparse.linalg import aslinearoperator, LinearOperator
	# # B = np.eye(A.shape[0]) - Q @ Q.T
	# # deflate_proj = lambda w: B @ (A @ (B @ w))
	# # L = LinearOperator(matvec = deflate_proj, shape=A.shape)
	# from primate.operator import OrthComplement
	# L = OrthComplement(A, Q)

	# from primate.trace import hutch 
	# true_residual = A.trace() - bulk_tr
	# np.abs((bulk_tr + hutch(L, atol = 0.10, maxiter=1000, verbose=True)) - A.trace())
	# hutch(L)
	# hutch(L, atol = 0.001, maxiter=200, deg=40, verbose=True, plot=True)
	# 0.9534050697745574 - 0.003 <= true_residual
	# true_residual <= 0.9534050697745574 + 0.003

	# residual_tr = 0.0
	# if True: 
	# 	G = W1 - Q @ Q.T @ W1
	# 	residual_tr = (1 / (m // 3)) * (G.T @ (A @ G)).trace()

	# print(f"A trace: {A.trace():8f}")
	# for i in range(1, A.shape[0]):
	# 	np.random.seed(1234)
	# 	est = hutch_pp(A, m = 3 * i)
	# 	print(f"{i}: {est} (error: {np.abs(est - A.trace())})")