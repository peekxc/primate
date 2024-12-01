from functools import partial
from typing import Callable, Union, Optional

import numpy as np
import scipy as sp  # allows for lazy loading

_ISO_DISTRIBUTIONS = {
	"rademacher": "rademacher",
	"normal": "normal",
	"sphere": "sphere",
	"signs": "rademacher",
	"gaussian": "normal",
}


def symmetric(
	n: int,
	dist: str = "normal",
	pd: bool = True,
	ew: Optional[np.ndarray] = None,
	seed: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
	"""Generates a random symmetric matrix of size `n` with eigenvalues `ew`.

	Parameters:
		n: The size of the matrix.
		dist: Distribution of individual matrix entries.
		pd: Whether to ensure the generated matrix is positive-definite, clipping eigenvalues as necessary.
		ew: Desired eigenvalues of `A`. If not provided, generates random values in the range [-1, 1].
		seed: seed for the random number generator.

	Returns:
		A random symmetric matrix with the presribed eigenvalues.
	"""
	rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
	N: int = n * (n - 1) // 2
	if dist == "uniform":
		A = sp.spatial.distance.squareform(rng.uniform(size=N))
		np.einsum("ii->i", A)[:] = rng.random(n)
	elif dist == "normal":
		A = sp.spatial.distance.squareform(rng.normal(size=N))
		np.einsum("ii->i", A)[:] = rng.random(n)
	else:
		raise ValueError(f"Invalid distribution {dist} supplied")
	Q, R = np.linalg.qr(A)
	if ew is None:
		ew = rng.uniform(size=n, low=0.0 if pd else -1.0, high=1.0)
	ew = np.atleast_1d(ew)
	A = Q @ np.diag(ew) @ Q.T
	A = (A + A.T) / 2
	return A


def haar(n: int, ew: Optional[np.ndarray] = None, seed: Union[int, np.random.Generator, None] = None) -> np.ndarray:
	"""Generates a random matrix with prescribed eigenvalues by sampling uniformly from the orthogonal group O(n).

	Parameters:
		n: The size of the matrix.
		ew: Desired eigenvalues of `A`. If not provided, generates random values in the range [0, 1].
		seed: seed for the random number generator.

	Returns:
		A random matrix with the presribed eigenvalues.
	"""
	rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
	OG = sp.stats.ortho_group(n, seed=rng)
	ew = rng.uniform(size=n, low=-1.0, high=1.0) if ew is None else np.atleast_1d(ew)
	assert len(ew) == n, "Number of eigenvalues must be <= `n`"
	ev = np.zeros(n)
	ev[: len(ew)] = ew
	U = OG.rvs()
	return U @ np.diag(ev) @ U.T


def isotropic(
	size: Union[int, tuple, None] = None, pdf: str = "rademacher", seed: Union[int, np.random.Generator, None] = None
) -> Union[np.ndarray, Callable]:
	"""Generates random vectors from a specified isotropic distribution.

	Parameters:
		size: Output shape to generate.
		pdf: Isotropic distribution to sample from. Must be "rademacher", "sphere", or "normal".
		seed: Seed or generator for pseudorandom number generation.

	Returns:
		Array of shape `size` with rows distributed according to `pdf`.
	"""
	assert pdf in _ISO_DISTRIBUTIONS.keys(), f"Invalid distribution '{pdf}' supplied."
	pdf: str = _ISO_DISTRIBUTIONS[pdf]
	rng = np.random.default_rng(seed)
	if size is None:
		return partial(isotropic, pdf=pdf, seed=rng)
	size = (1, size) if isinstance(size, int) else size
	if pdf == "rademacher":
		W = rng.choice([-1.0, +1.0], size=size, replace=True)
		# W /= np.repeat(np.sqrt(m), n)[:, np.newaxis]
	elif pdf == "sphere":
		# "On the real sphere with radius sqrt(m)"
		# https://mathoverflow.net/questions/24688/efficiently-sampling-points-uniformly-from-the-surface-of-an-n-sphere
		W = rng.normal(size=size, loc=0.0, scale=1.0)
		# W /= np.linalg.norm(W, axis=1)[:, np.newaxis]
		W /= np.sqrt(np.sum(W**2, axis=-1, keepdims=True))
		W *= np.sqrt(W.shape[-1])
	else:
		W = rng.normal(size=size, loc=0.0, scale=1.0)
	return W
