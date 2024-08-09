from typing import Union, Callable, Any
from numbers import Integral
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve_triangular
from scipy.stats import t
from scipy.special import erfinv
from numbers import Real, Number

## Package imports
from .random import _engine_prefixes, _engines, isotropic
from .special import _builtin_matrix_functions
from .operator import matrix_function
import _lanczos
import _trace
import _orthogonalize


def sl_gauss(
	A: Union[LinearOperator, np.ndarray],
	n: int = 150,
	deg: int = 20,
	pdf: str = "rademacher",
	rng: str = "pcg",
	seed: int = -1,
	orth: int = 0,
	num_threads: int = 0,
) -> np.ndarray:
	"""Stochastic Gaussian quadrature approximation.

	Computes a set of sample nodes and weights for the degree-k orthogonal polynomial approximating the 
	cumulative spectral measure of `A`. This function can be used to approximate the spectral density of `A`, 
	or to approximate the spectral sum of any function applied to the spectrum of `A`.

	Parameters
	----------
	A : ndarray, sparray, or LinearOperator
	    real symmetric operator.
	n : int, default=150
	    Number of random vectors to sample for the quadrature estimate.
	deg  : int, default=20
	    Degree of the quadrature approximation.
	rng : { 'splitmix64', 'xoshiro256**', 'pcg64', 'lcg64', 'mt64' }, default="pcg64"
	    Random number generator to use (PCG64 by default).
	seed : int, default=-1
	    Seed to initialize the `rng` entropy source. Set `seed` > -1 for reproducibility.
	pdf : { 'rademacher', 'normal' }, default="rademacher"
	    Choice of zero-centered distribution to sample random vectors from.
	orth: int, default=0
		Number of additional Lanczos vectors to orthogonalize against when building the Krylov basis.
	num_threads: int, default=0
	    Number of threads to use to parallelize the computation. Setting `num_threads` < 1 to let OpenMP decide.

	Returns
	-------
	trace_estimate : float
	    Estimate of the trace of the matrix function $f(A)$.
	info : dict, optional
	    If 'info = True', additional information about the computation.

	"""
	attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
	assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method"
	assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
	assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"

	## Choose the random number engine
	assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
	engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)

	## Choose the distribution to sample random vectors from
	assert pdf in ["rademacher", "normal"], f"Invalid distribution '{pdf}'; Must be one of 'rademacher' or 'normal'."
	distr_id = ["rademacher", "normal"].index(pdf)

	## Get the dtype; infer it if it's not available
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert (
		f_dtype.type == np.float32 or f_dtype.type == np.float64
	), "Only 32- or 64-bit floating point numbers are supported."

	## Extract the machine precision for the given floating point type
	lanczos_rtol = np.finfo(f_dtype).eps  # if lanczos_rtol is None else f_dtype.type(lanczos_rtol)

	## Argument checking
	m = A.shape[1]  # Dimension of the space
	nv = int(n)  # Number of random vectors to generate
	seed = int(seed)  # Seed should be an integer
	deg = max(deg, 2)  # Must be at least 2
	orth = m - 1 if orth < 0 else min(m - 1, orth)  # Number of additional vectors should be an integer
	ncv = max(int(deg + orth), m)  # Number of Lanczos vectors to keep in memory
	num_threads = int(num_threads)  # should be integer; if <= 0 will trigger max threads on C++ side

	## Collect the arguments processed so far
	sl_quad_args = (nv, distr_id, engine_id, seed, deg, lanczos_rtol, orth, ncv, num_threads)

	## Make the actual call
	quad_nw = _lanczos.stochastic_quadrature(A, *sl_quad_args)
	return quad_nw
