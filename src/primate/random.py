import numpy as np
from typing import Union, Optional, Callable
import _random_gen
from math import prod
from numbers import Integral
from scipy.spatial.distance import squareform

_engines = ["splitmix64", "xoshiro256**", "pcg64", "mt64"]
_engine_prefixes = ["sx", "xs", "pcg", "mt"]
_iso_distributions = ["rademacher", "normal", "sphere", "xtrace"]

def symmetric(n: int, dist: str = "normal", pd: bool = True, ew: np.ndarray = None):
	N: int = n * (n - 1) // 2
	if dist == "uniform":
		A = squareform(np.random.uniform(size=N))
		np.einsum("ii->i", A)[:] = np.random.random(n)
	elif dist == "normal":
		A = squareform(np.random.normal(size=N))
		np.einsum("ii->i", A)[:] = np.random.random(n)
	else:
		raise ValueError(f"Invalid distribution {dist} supplied")
	ew = np.random.uniform(size=n, low=-1.0 + np.finfo(np.float32).eps, high=1.0) if ew is None else np.array(ew)
	if pd:
		ew = (ew + 1.0) / 2.0
	Q, R = np.linalg.qr(A)
	A = Q @ np.diag(ew) @ Q.T
	A = (A + A.T) / 2
	return A

def rademacher(size: Union[int, tuple], rng: str = "splitmix64", seed: int = -1, dtype=np.float32):
	"""Generates random vectors from the rademacher distribution.

	Parameters
	----------
	size : int or tuple
			Output shape to generate.
	rng : str = "splitmix64"
			Random number generator to use.
	seed : int = -1
			Seed for the generator. Use -1 to for random (non-deterministic) behavior.
	dtype : dtype = float32
			Floating point dtype for the output. Must be float32 or float64.
	"""
	assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
	assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
	engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)
	out = np.empty(size, dtype=dtype) if isinstance(size, Integral) else np.empty(prod(size), dtype=dtype)
	engine_f = getattr(_random_gen, "rademacher")
	engine_f(out, engine_id, seed)
	return out.reshape(size)


def normal(size: Union[int, tuple], rng: str = "splitmix64", seed: int = -1, dtype=np.float32):
	"""Generates random vectors from the rademacher distribution.

	Parameters
	----------
	size : int or tuple
			Output shape to generate.
	rng : str = "splitmix64"
			Random number generator to use.
	seed : int = -1
			Seed for the generator. Use -1 to for random (non-deterministic) behavior.
	dtype : dtype = float32
			Floating point dtype for the output. Must be float32 or float64.
	"""
	assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
	assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
	engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)
	out = np.empty(size, dtype=dtype) if isinstance(size, Integral) else np.empty(prod(size), dtype=dtype)
	engine_f = getattr(_random_gen, "normal")
	engine_f(out, engine_id, seed)
	return out.reshape(size)


# def rayleigh(n: int, engine: str = "splitmix64", num_threads: int = 1, dtype = np.float32):
#   assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
#   assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
#   engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
#   out = np.empty(n, dtype=dtype)
#   engine_f = getattr(_random_gen, 'rayleigh_'+_engine_prefixes[engine_id])
#   engine_f(out, num_threads)
#   return out


def isotropic(size: Union[int, tuple], method: str = "rademacher") -> np.ndarray:
	assert isinstance(method, str) and method in _iso_distributions, f"Invalid distribution '{str(method)}' supplied."
	n, m = (size, 1) if isinstance(size, Integral) else np.take(size, (0, 1))
	if method == "rademacher":
		W = rademacher(size=(n, m))
		W = W / np.repeat(np.sqrt(n), n)[:,np.newaxis]
		return W
	elif method == "sphere":
		W = np.random.normal(size=(n, m))
		W = np.sqrt(n) * (W / np.linalg.norm(W, axis=0))  ## confirmed
		return W
	else:
		raise ValueError("invalid method supplied")

