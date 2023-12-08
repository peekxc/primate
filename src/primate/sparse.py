import numpy as np 
from typing import Union
from scipy.sparse.linalg import LinearOperator
from .diagonalize import lanczos

class MatrixFunction(LinearOperator):
  """Approximates the action v |-> f(A)v """
  def __init__(self, lo: Union[np.ndarray, LinearOperator], fun: Callable[float, float], dtype = None):
    assert isinstance(lo, LinearOperator), "Must pass a linear operator"
    self.op = lo
    self.fun: Callable[float, float] = fun

  def _matvec(self, v: np.ndarray) -> np.ndarray:
    """ Matrix-vector multiplication (forward mode)"""
    pass

  def shape(self) -> tuple:
    return self.op.shape()