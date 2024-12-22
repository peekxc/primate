from functools import partial
from numbers import Integral
from typing import Callable, Union, Optional
import multiprocessing
import concurrent.futures

import numpy as np
from numpy.random import SeedSequence
import scipy as sp  # allows for lazy loading


_ISO_DISTRIBUTIONS = {
	"rademacher": "rademacher",
	"normal": "normal",
	"sphere": "sphere",
	"signs": "rademacher",
	"gaussian": "normal",
}


## NOTE: It's 2-3x faster to use the four ufuncs below than sign
def _rademacher_inplace(rng: np.random.Generator, out: np.ndarray) -> None:
	rng.random(out=out)
	# np.subtract(out, 0.5, out=out)
	# np.sign(out, out=out)
	np.multiply(out, 2, out=out)  # [0, 2]
	np.floor(out, out=out)  # {0, 1}
	np.multiply(out, 2, out=out)  # {0, 2}
	np.subtract(out, 1, out=out)  # {-1, +1}


def _normal_inplace(rng: np.random.Generator, out: np.ndarray) -> None:
	rng.standard_normal(out=out, dtype=out.dtype)


def _sphere_inplace(rng: np.random.Generator, out: np.ndarray) -> None:
	rng.standard_normal(out=out, dtype=out.dtype)
	c = np.sqrt(np.sum(out**2, axis=0, keepdims=True))  # or np.sqrt(np.einsum('ij,ij -> j', Y, Y))
	n = np.sqrt(out.shape[0])
	np.divide(out, c, out=out)
	np.multiply(out, n, out=out)


_ISO_FUNCS = {"rademacher": _rademacher_inplace, "sphere": _sphere_inplace, "normal": _normal_inplace}


def isotropic(
	size: Union[int, tuple, None] = None,
	pdf: str = "rademacher",
	seed: Union[int, np.random.Generator, None] = None,
	out: Optional[np.ndarray] = None,
) -> Union[None, np.ndarray, Callable]:
	"""Generates random vectors from a specified isotropic distribution.

	Parameters:
		size: Output shape to generate.
		pdf: Isotropic distribution to sample from. Must be "rademacher", "sphere", or "normal".
		seed: Seed or generator for pseudorandom number generation.
		out: Output array to fill values in-place.

	Returns:
		Array of shape `size` with rows distributed according to `pdf`.
	"""
	assert pdf in _ISO_DISTRIBUTIONS.keys(), f"Invalid distribution '{pdf}' supplied."
	pdf: str = _ISO_DISTRIBUTIONS[pdf]
	rng = np.random.default_rng(seed)
	if out is not None:
		assert isinstance(out, np.ndarray)
		_ISO_FUNCS[pdf](rng, out)
		return None
	else:
		iso = _ISO_FUNCS[pdf]

		def _isotropic(size: Union[int, tuple]):
			size = (size, 1) if isinstance(size, int) else size
			W = np.empty(shape=size, dtype=np.float64, order="F")
			iso(rng, out=W)
			return W

		return _isotropic if size is None else _isotropic(size)

		# W = np.empty(shape=size, dtype=np.float64, order="C")
		# _ISO_FUNCS[pdf](rng=rng, out=W)
		# return W
	# if pdf == "rademacher":
	# 	W = rng.choice([-1.0, +1.0], size=size, replace=True)
	# 	# W /= np.repeat(np.sqrt(m), n)[:, np.newaxis]
	# elif pdf == "sphere":
	# 	# "On the real sphere with radius sqrt(m)"
	# 	# https://mathoverflow.net/questions/24688/efficiently-sampling-points-uniformly-from-the-surface-of-an-n-sphere
	# 	W = rng.normal(size=size, loc=0.0, scale=1.0)
	# 	# W /= np.linalg.norm(W, axis=1)[:, np.newaxis] # this has a runtime cost
	# 	W /= np.sqrt(np.sum(W**2, axis=-1, keepdims=True))
	# 	W *= np.sqrt(W.shape[-1])
	# else:
	# 	W = rng.normal(size=size, loc=0.0, scale=1.0)
	# return W


class Isotropic:
	def __init__(
		self,
		size: tuple,
		pdf: str = "signs",
		seed: Union[int, np.random.SeedSequence, np.random.Generator, None] = None,
		threads: Optional[int] = None,
	):
		pdf: str = _ISO_DISTRIBUTIONS[pdf]
		assert pdf in _ISO_DISTRIBUTIONS.keys(), f"Invalid distribution '{pdf}' supplied."
		self.pdf = _ISO_DISTRIBUTIONS[pdf]
		self.iso = _ISO_FUNCS[pdf]
		self.threads = multiprocessing.cpu_count() if threads is None else int(threads)
		rng = np.random.default_rng(seed)  # type: ignore
		assert isinstance(rng, np.random.Generator) and hasattr(rng, "spawn")
		self._random_generators = [rng] if self.threads == 1 else rng.spawn(self.threads)
		# if isinstance(seed, np.random.BitGenerator) or seed is None:
		# 	rng = np.random.default_rng(seed)  # type: ignore
		# elif isinstance(seed, Integral) or isinstance(seed, SeedSequence):
		# 	rng = np.random.default_rng(seed)
		# else:
		# 	raise ValueError("Unknown type 'seed'")

		# if isinstance(seed, np.random.BitGenerator) or seed is None:
		# else:
		# 	seq = np.random.SeedSequence(seed) if isinstance(seed, Integral) else seed  # type: ignore
		# 	assert isinstance(seed, SeedSequence), "Custom seed for multiple threads must be integer or SeedSequence."
		# 	self._random_generators = [np.random.default_rng(s) for s in seq.spawn(self.threads)]  # type: ignore
		self.shape = size
		self.executor = concurrent.futures.ThreadPoolExecutor(self.threads)
		self.values = np.zeros(size, order="F")  # NOTE: this needs to be fortran order for multithreaded
		self.step = np.ceil(self.shape[1] / self.threads).astype(np.int_)

	def fill(self):
		_fill = lambda rng, out, i1, i2: self.iso(rng, out=out[:, i1:i2])  # noqa: E731
		futures = {}
		for i in range(self.threads):
			args = (_fill, self._random_generators[i], self.values, i * self.step, (i + 1) * self.step)
			futures[self.executor.submit(*args)] = i
		concurrent.futures.wait(futures)

	def __del__(self):
		self.executor.shutdown(False)


def symmetric(
	n: int,
	dist: str = "normal",
	pd: bool = False,
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
