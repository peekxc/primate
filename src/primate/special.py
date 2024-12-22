import typing
from typing import Any, Callable, Optional, Union

import numpy as np

## Natively support matrix functions
_BUILTIN_MATRIX_FUNCTIONS = ["identity", "abs", "sqrt", "log", "inv", "exp", "smoothstep", "numrank"]


def softsign(x: Optional[np.ndarray] = None, q: int = 1) -> Union[Callable, np.ndarray]:
	"""Softer (smoother) variant of the discontinuous sign function.

	This function computes a continuous version of the sign function (centered at 0) which is uniformly close to the
	sign function for sufficiently large q, and converges to sgn(x) as q -> +infty for all x in [-1, 1].

	The soft-sign function is often used in principal component regression and norm estimation algorithms, see
	equation (60) of "Stability of the Lanczos Method for Matrix Function Approximation"
	"""
	I = np.arange(q + 1)  # noqa: E741
	J = np.append([1], np.cumprod([(2 * j - 1) / (2 * j) for j in np.arange(1, q + 1)]))

	def _sign(x: Union[float, np.ndarray]) -> np.ndarray:
		x = np.clip(x, -1.0, +1.0)
		x = np.atleast_2d(x).T
		sx = np.ravel(np.sum(x * (1 - x**2) ** I * J, axis=1))
		return sx

	return _sign(x) if x is not None else _sign


def smoothstep(
	x: Optional[np.ndarray] = None, a: float = 0.0, b: float = 1.0, deg: int = 3
) -> Union[Callable, np.ndarray]:
	"""Smoothstep function.

	This function computes a continuous version of the standard 'step' function by cubic Hermite
	interpolation "sigmoid-like" curve between 0 and 1 on the domain [a,b].

	The smoothstep function is often used in computer graphics and in shader programming, see the wikipedia
	page "smoothstep" for more details. Also see [this video](https://www.youtube.com/watch?v=60VoL-F-jIQ)
	for desmos visualization that derives it.
	"""
	assert (deg % 2) == 1, "Degree must be odd"
	d: float = (b - a) if a != b else 1.0

	def _smoothstep(x):
		y = np.clip((x - a) / d, 0.0, 1.0)  # maps [a,b] |-> [0,1]
		y = 3 * y**2 - 2 * y**3
		return y

	if x is not None:
		return _smoothstep(x)
	return _smoothstep
	# Vectorize just has way too much overhead
	# return np.vectorize(np.frompyfunc(_smoothstep, nin=1, nout=1, identity=0), otypes=["d"])


def identity(x: Any) -> Any:
	return x


def exp(x: Optional[np.ndarray] = None, t: float = 1.0) -> Union[Callable, np.ndarray]:
	def _exp(x):
		return np.exp(t * x)

	return _exp(x) if x is not None else _exp


def step(x: Optional[np.ndarray] = None, c: float = 0.0, nonnegative: bool = False) -> Union[Callable, np.ndarray]:
	def _step(x):
		x = np.abs(x) if nonnegative else x
		return np.where(x < c, 0.0, 1.0)

	return _step(x) if x is not None else _step


@typing.no_type_check
def param_callable(fun: Union[str, None], **kwargs) -> Callable:
	if isinstance(fun, str):
		assert fun in _BUILTIN_MATRIX_FUNCTIONS, "If given as a string, matrix_function be one of the builtin functions."
	if fun is None or fun == "identity":
		return identity
	elif callable(fun):
		return fun
	elif fun == "abs":
		return np.abs
	elif fun == "sqrt":
		return np.sqrt
	elif fun == "log":
		return lambda x: np.log(np.maximum(x, np.finfo(np.float64).eps))
	elif fun == "inv":
		return np.reciprocal
	elif fun == "exp":
		t = kwargs.pop("t", 1.0)  # type: ignore
		return exp(t=t)
	elif fun == "smoothstep":
		a = kwargs.pop("a", 0.0)  # type: ignore
		b = kwargs.pop("b", 1.0)  # type: ignore
		return smoothstep(a=a, b=b)
	elif fun == "softsign":
		q = kwargs.pop("q", 10)  # type: ignore
		return softsign(q=q)
	elif fun == "numrank":
		threshold = kwargs.pop("threshold", 0.000001)  # type: ignore
		return step(c=threshold, nonnegative=True)
	else:
		raise ValueError(f"Unknown function: {fun}.")
