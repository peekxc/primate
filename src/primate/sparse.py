import numpy as np 
from typing import Union
from scipy.sparse.linalg import LinearOperator
import _lanczos


def matrix_function(
  A: Union[LinearOperator, np.ndarray],
  fun: Union[str, Callable] = "identity", 
  deg: int = 20,
  orth: int = 0
) -> LinearOperator:
  """Constructs an operator approximating the action v |-> f(A)v
  
  This function constructs an operator that uses the Lanczos method to approximates the action of 
  matrix-vector multiplication with the matrix function f(A) for some choice of `fun`.

  Parameters:
  ----------- 
    A : ndarray, sparray, or LinearOperator
        real symmetric operator.
    fun : str or Callable, default = "identity"
        real-valued function defined on the spectrum of `A`.
    deg : int, default = 20
        Degree of the Krylov expansion.
    orth: int, default = 0
        Number of additional Lanczos vectors to orthogonalize against when building the Krylov basis.
    
  """
  pass

# class MatrixFunction(LinearOperator):
#   """Approximates the action v |-> f(A)v """
#   def __init__(self, lo: Union[np.ndarray, LinearOperator], fun: Callable[float, float], dtype = None):
#     assert isinstance(lo, LinearOperator), "Must pass a linear operator"
#     self.op = lo
#     self.fun: Callable[float, float] = fun

#   def _matvec(self, v: np.ndarray) -> np.ndarray:
#     """ Matrix-vector multiplication (forward mode)"""
#     pass

#   def shape(self) -> tuple:
#     return self.op.shape()