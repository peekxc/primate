import numpy as np
from typing import Union, Callable
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse

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
	"""Constructs an operator approximating the action $v \mapsto f(A)v$

	This function constructs an operator that uses the Lanczos quadrature method to approximate the action of
	matrix-vector multiplication with the matrix function $f(A)$ for any choice of `fun`.

	Parameters:
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

	Returns:
	--------
	  operator : LinearOperator
	      Operator approximating the action of `fun` on the spectrum of `A`

	Notes: 
	------
	The matrix-function approximation is implemented via Lanczos quadrature, which combines the Lanczos method with the 
	Golub-Welsch algorithm for computing the `deg`-point Gaussian quadrature rule of a symmetric tridiagonal / Jacobi 
	matrix. The underlying tridiagonal eigen-computation uses implicit symmetric QR steps with Wilkinson shifts.

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
