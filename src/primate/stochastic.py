import numpy as np
from typing import Union, Optional, Callable
from scipy.spatial.distance import squareform

_ISO_DISTRIBUTIONS = {"rademacher", "normal", "sphere"}


def symmetric(
	n: int,
	dist: str = "normal",
	pd: bool = True,
	ew: np.ndarray = None,
	seed: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
	"""Generates a random symmetric matrix of size `n` with eigenvalues `ew`.

	Parameters:
		n: The size of the matrix.
		dist: Distribution of individual matrix entries.
		pd: Whether to ensure the generated matrix is positive-definite. Potentially clips eigenvalues.
		ew: Desired eigenvalues of `A`. If not provided, generates random values in the range [0, 1].

	Returns:
		A random symmetric matrix with the presribed eigenvalues.
	"""
	rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
	N: int = n * (n - 1) // 2
	if dist == "uniform":
		A = squareform(rng.uniform(size=N))
		np.einsum("ii->i", A)[:] = rng.random(n)
	elif dist == "normal":
		A = squareform(rng.normal(size=N))
		np.einsum("ii->i", A)[:] = rng.random(n)
	else:
		raise ValueError(f"Invalid distribution {dist} supplied")
	if ew is None:
		ew = rng.uniform(size=n, low=0.0 if pd else -1.0, high=1.0)
	else:
		ew = np.asarray(ew)
	Q, R = np.linalg.qr(A)
	A = Q @ np.diag(ew) @ Q.T
	A = (A + A.T) / 2
	return A


def isotropic(
	size: Union[int, tuple], method: str = "rademacher", seed: Union[int, np.random.Generator, None] = None
) -> np.ndarray:
	"""Generates random vectors from a specified isotropic distribution.

	Parameters:
		size: Output shape to generate.
		method: Isotropic distribution to sample from. Must be "rademacher", "sphere", or "normal".
		seed: Seed or generator for pseudo random number generation.

	Returns:
		Randomly generated array of shape `size` with entries distributed according to `method`.
	"""
	assert isinstance(method, str) and method in _ISO_DISTRIBUTIONS, f"Invalid distribution '{method}' supplied."
	rng = np.random.default_rng(seed)
	n, m = (1, size) if isinstance(size, int) else np.take(size, (0, 1))
	if method == "rademacher":
		W = rng.choice([-1.0, +1.0], size=(n, m), replace=True)
		# W /= np.repeat(np.sqrt(m), n)[:, np.newaxis]
	elif method == "sphere":
		# "On the real sphere with radius sqrt(m)"
		# https://mathoverflow.net/questions/24688/efficiently-sampling-points-uniformly-from-the-surface-of-an-n-sphere
		W = rng.normal(size=(n, m), loc=0.0, scale=1.0)
		W /= np.linalg.norm(W, axis=1)[:, np.newaxis]
		W *= np.sqrt(m)
	else:
		W = rng.normal(size=(n, m), loc=0.0, scale=1.0)
	return W
