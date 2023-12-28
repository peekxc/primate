import numpy as np
from typing import Union, Callable
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse
from numpy.typing import ArrayLike

from .special import _builtin_matrix_functions
import _operators

def matrix_function(
	A: Union[LinearOperator, np.ndarray],
	fun: Union[str, Callable] = "identity",
	deg: int = 20,
	rtol: float = None,
	orth: int = 0,
	**kwargs,
) -> LinearOperator:
	"""Constructs a `LinearOperator` approximating the action of the matrix function `fun(A)`. 

	This function uses the Lanczos quadrature method to approximate the action of matrix function:
	$$ v \mapsto f(A)v, \quad f(A) = U f(\lambda) U^T $$
	The resulting operator may be used in conjunction with other trace and vector-based estimation methods, 
	such as the `hutch` or `xtrace` estimators.

	The weights of the quadrature may be computed using either the Golub-Welsch (GW) or 
	Forward Three Term Recurrence algorithms (FTTR) (see the `quad` parameter). For a description 
	of the other parameters, see the Lanczos function. 

	:::{.callout-note}
	To compute the weights of the quadrature, the GW computation uses implicit symmetric QR steps with Wilkinson shifts, 
	while the FTTR algorithm uses the explicit expression for orthogonal polynomials. While both require $O(\\mathrm{deg}^2)$ time to execute, 
	the former requires $O(\mathrm{deg}^2)$ space but is highly accurate, while the latter uses only $O(1)$ space at the cost of stability. 
	If `deg` is large, `fttr` is preferred. 
	:::

	Parameters
	-----------
		A : ndarray, sparray, or LinearOperator
				real symmetric operator.
		fun : str or Callable, default="identity"
				real-valued function defined on the spectrum of `A`.
		deg : int, default=20
				Degree of the Krylov expansion.
		rtol : float, default=1e-8
				Relative tolerance to consider two Lanczos vectors are numerically orthogonal.
		orth: int, default=0
				Number of additional Lanczos vectors to orthogonalize against when building the Krylov basis.
		kwargs : dict, optional
			additional key-values to parameterize the chosen function 'fun'.

	Returns
	-------
	: 
		a `LinearOperator` approximating the action of `fun` on the spectrum of `A`

	"""
	attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
	assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method"
	assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
	assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"

	## Parameterize the type of matrix function
	if isinstance(A, np.ndarray):
		module_func = "Dense"
	elif issparse(A):
		module_func = "Sparse"
	elif isinstance(A, LinearOperator):
		module_func = "Generic"
	else:
		raise ValueError(f"Invalid type '{type(A)}' supplied for operator A")

	## Get the dtype; infer it if it's not available
	f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
	assert (f_dtype.type == np.float32 or f_dtype.type == np.float64), "Only 32- or 64-bit floating point numbers are supported."
	module_func += "F" if f_dtype.type == np.float32 else "D"

	## Argument checking
	rtol = np.finfo(f_dtype).eps if rtol is None else f_dtype.type(rtol)
	orth = np.clip(orth, 0, deg)
	deg = np.clip(deg, 2, A.shape[0])   # Should be at least two

	## Parameterize the matrix function and trace call
	if fun is None: 
		kwargs["function"] = "None"
	elif isinstance(fun, str):
		assert fun in _builtin_matrix_functions, f"If given as a string, 'fun' must be one of {str(_builtin_matrix_functions)}."
		kwargs["function"] = fun  # _builtin_matrix_functions.index(matrix_function)
	elif isinstance(fun, Callable):
		kwargs["function"] = "generic"
		kwargs["matrix_func"] = fun
	else:
		raise ValueError(f"Invalid matrix function type '{type(fun)}'")

	## Construct the instance
	module_func += "_MatrixFunction"
	M = getattr(_operators, module_func)(A, deg, rtol, orth, deg, **kwargs)
	return M

## From: https://www.mathworks.com/matlabcentral/fileexchange/8548-toeplitzmult
class Toeplitz(LinearOperator):
	def __init__(self, c: ArrayLike, r: ArrayLike = None, dtype = None):
		self.c = np.array(c)
		self.r = np.array(c if r is None else r)
		self._d = np.concatenate((self.c,[0],np.flip(self.r[1:])))
		self._dfft = np.real(np.fft.fft(self._d))
		self._z = np.zeros(len(c)*2) # workspace
		self.shape = (len(c), len(c))
		self.dtype = np.dtype(np.float64) if dtype is None else dtype
	
	def _matvec(self, x: np.ndarray) -> np.ndarray:
		assert len(x) == len(self.c), f"Invalid shape of input vector 'x'; must have length {len(self.c)}"
		self._z[:len(x)] = x
		x_fft = np.fft.fft(self._z)
		y = np.real(np.fft.ifft(self._dfft * x_fft))
		return y[:len(x)]