import numpy as np
from typing import Union
from scipy.sparse.linalg import LinearOperator
from numbers import Number
from scipy.linalg import eigvalsh_tridiagonal

from .trace import hutch, xtrace
from .operator import matrix_function
from .diagonalize import lanczos

def numrank(
  A: Union[LinearOperator, np.ndarray],
	est: str = "hutch",
  eps: Union[float, str] = "auto",
	**kwargs
):
  ## Use lanczos to get basic estimation of smallest positive eigenvalue
  # if eps == "auto":
  a,b = lanczos(A, deg=A.shape[0])
  rr = np.sort(eigvalsh_tridiagonal(a,b))
  EPS = np.finfo(A.dtype).eps
  tol = np.max(rr) * A.shape[0] * EPS # NumPy default tolerance 
  gap = np.min(rr[rr > tol])
  # else: 

  # rr_mag = np.maximum(np.abs(rr[:-1]), EPS)
  # max_rel_ind = np.argmax(np.diff(rr) / rr_mag) + 1
  # tol = rr[max_rel_ind]

  # assert min_ew_est > 10 * EPS, f"Estimated smallest positive eigenvalue is too small: {min_ew_est}"

  ## Estimate numerical rank by w/ Hutch
  if est == "hutch":
    N = 10 * A.shape[0]
    default_kwargs = dict(maxiter=N, threshold=0.95*tol, atol=0.50, info=False, verbose=False)
    default_kwargs.update(kwargs)
    est = hutch(A, fun="numrank", **default_kwargs)
  elif est == "xtrace":
    M = matrix_function(A, fun="smoothstep", a=tol, b=gap)
    est = xtrace(M)
  return int(np.round(est)) if isinstance(est, Number) else est