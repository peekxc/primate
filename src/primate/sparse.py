import numpy as np 
from typing import Union
from scipy.sparse.linalg import LinearOperator
from .diagonalize import lanczos

class MatrixFunction(LinearOperator):
  def __init__(lo: Union[np.ndarray, LinearOperator], fun: Callable[float, float]):
    assert isinstance(lo, LinearOperator), "Must pass a linear operator"
    self.op = lo
    self.fun = fun

  def _matvec():


