import numpy as np
from typing import * 
from scipy.sparse.linalg import LinearOperator
from numbers import Number
from primate.trace import hutch, xtrace
from primate.operator import matrix_function
from primate.diagonalize import lanczos
from scipy.linalg import eigvalsh_tridiagonal

def numrank(
  A: Union[LinearOperator, np.ndarray],
	est: str = "hutch",
	**kwargs
):
  ## Use lanczos to get basic estimation of smallest positive eigenvalue
  a,b = lanczos(A, deg=A.shape[0])
  rr = np.sort(eigvalsh_tridiagonal(a,b))
  EPS = np.finfo(A.dtype).eps
  # rr_mag = np.maximum(np.abs(rr[:-1]), EPS)
  # max_rel_ind = np.argmax(np.diff(rr) / rr_mag) + 1
  # tol = rr[max_rel_ind]
  tol = np.max(rr) * A.shape[0] * EPS
  # assert min_ew_est > 10 * EPS, f"Estimated smallest positive eigenvalue is too small: {min_ew_est}"

  ## Estimate numerical rank by w/ Hutch
  if est == "hutch":
    N = 10 * A.shape[0]
    default_kwargs = dict(maxiter=N, threshold=0.95*tol, atol=0.50, info=False, verbose=False)
    est = hutch(A, fun="numrank", **(default_kwargs | kwargs))
  elif est == "xtrace":
    M = matrix_function(A, fun="numrank", threshold=0.95*tol)
    est = xtrace(M)
  return int(np.round(est)) if isinstance(est, Number) else est