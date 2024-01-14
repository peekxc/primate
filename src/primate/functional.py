import numpy as np
from typing import Union
from scipy.sparse.linalg import LinearOperator
from numbers import Number
from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal
import bisect

from .trace import hutch, xtrace
from .operator import matrix_function
from .diagonalize import lanczos

## Since Python has no support for sized generators 
class RelativeErrorBound():
  def __init__(self, n: int):
    self.base_num = 2.575 * np.log(n)
    self._len = n 

  def __len__(self):
    return self._len

  def __getitem__(self, i: int):
    return -self.base_num / i**2

def numrank(
  A: Union[LinearOperator, np.ndarray],
	est: str = "hutch",
  gap: Union[float, str] = "auto",
  gap_rtol: float = 0.01, 
  psd: bool = True,
	**kwargs
):
  """Estimates the numerical rank of a given operator via stochastic trace estimation. 
  
  Parameters
  ----------

  est : str 
      The trace estimator to use.  
  gap : str or float 

  """
  ## Use lanczos to get basic estimation of largest and smallest positive eigenvalues
  ## Relative error bounds based on: the Largest Eigenvalue by the Power and Lanczos Algorithms with a Random Starts
  ## Spectral gap bounds based on sec. 13.2 of "The Symmetric Eigenvalue Problem" by Paige and the by 
  ## comments by Lin in "APPROXIMATING SPECTRAL DENSITIES OF LARGE MATRICES"
  default_kwargs = {}
  if gap == "auto" or gap == "simple":
    EPS = np.finfo(A.dtype).eps
    deg = max(kwargs.get("deg", 20), 4)
    n = A.shape[0]
    if n < 150:
      rel_error_bound = 2.575 * np.log(A.shape[0]) / np.arange(4, n)**2
      deg_bound = max(np.searchsorted(-rel_error_bound, -0.01) + 5, deg)
    else: 
      ## This does binary search like searchsorted but uses O(1) memory
      re_bnd = RelativeErrorBound(n)
      deg_bound = max(bisect.bisect_left(re_bnd, -0.01) + 1, deg)
    
    ## Use PSD-specific theory to estimate spectral gap 
    a,b = lanczos(A, deg=deg_bound)
    if psd:  
      if gap == "auto":   
        rr, rv = eigh_tridiagonal(a,b)
        tol = np.max(rr) * A.shape[0] * EPS # NumPy default tolerance 
        min_id = np.flatnonzero(rr >= tol)[np.argmin(rr[rr >= tol])] # [0,n]
        coeff = b[min_id-1] if min_id == len(b) else min([b[min_id-1], b[min_id]])
        gap = rr[min_id] - coeff * np.abs(rv[-1,min_id])
        gap = rr[min_id] if gap < 0 else gap
      elif gap == "simple":      
        ## This is typically a better estimate of the gap, but has little theory 
        rr = eigvalsh_tridiagonal(a,b)
        tol = np.max(rr) * A.shape[0] * EPS
        denom = np.where(rr[:-1] == 0, 1.0, rr[:-1])
        gap = max(rr[np.argmax(np.diff(rr) / denom) + 1], tol)
    else: 
      rr = eigvalsh_tridiagonal(a,b)
      gap = np.max(rr) * A.shape[0] * EPS # NumPy default tolerance 
      tol = A.shape[0] * EPS
    default_kwargs.update(dict(fun="smoothstep", a=tol, b=gap))
  else: 
    assert isinstance(gap, Number), "Threshold `eps` must be a number"
    default_kwargs.update(dict(fun="numrank", threshold=gap))

  ## Estimate numerical rank
  if est == "hutch":
    ## By default, estimate numerical rank to within rounding accuracy
    ## Caps the number of the iterations using SciPy / ARPACK's heuristic
    N = 10 * A.shape[0] ## scipy's default for eigsh 
    default_kwargs.update(dict(maxiter=N, atol=0.50))
    default_kwargs.update(kwargs)
    est = hutch(A, **default_kwargs)
  elif est == "xtrace":
    default_kwargs.update(kwargs)
    M = matrix_function(A, **default_kwargs)
    est = xtrace(M)
  return int(np.round(est)) if isinstance(est, Number) else est