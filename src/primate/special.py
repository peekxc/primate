from typing import *
import numpy as np 

## Natively support matrix functions
_builtin_matrix_functions = ["identity", "abs", "sqrt", "log", "inv", "exp", "smoothstep", "numrank", "gaussian"]

def soft_sign(x: np.ndarray = None, q: int = 1):
  """Soft-sign function.
  
  This function computes a continuous version of the sign function (centered at 0) which is uniformly close to the 
  sign function for sufficiently large q, and converges to sgn(x) as q -> +infty for all x in [-1, 1].  

  The soft-sign function is often used in principal component regression and norm estimation algorithms, see 
  equation (60) of "Stability of the Lanczos Method for Matrix Function Approximation"
  """
  I = np.arange(q+1)
  J = np.append([1], np.cumprod([(2*j-1)/(2*j) for j in np.arange(1, q+1)]))
  def _sign(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x = np.clip(x,-1.0,+1.0)
    x = np.atleast_2d(x).T
    sx = np.ravel(np.sum(x * (1-x**2)**I * J, axis=1))
    return sx if len(sx) > 1 else np.take(sx,0)
  return _sign(x) if x is not None else _sign




