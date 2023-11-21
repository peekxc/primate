import numpy as np 
from typing import *
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
import _lanczos 

def lanczos(
  A: LinearOperator, 
  v0: Optional[np.ndarray] = None, 
  max_steps: int = None, 
  tol: float = 1e-8, 
  orth: int = 0, 
  sparse_mat: bool = False, 
  return_basis: bool = False, 
  seed: int = None, 
  dtype = None
):
  """Lanczos method of minimized iterations.

  Parameters
  ----------
  A : LinearOperator | sparray
    Symmetric operator to tridiagonalize. 
  v0 : ndarray, default = None
    Initial vector to orthogonalize against.
  max_steps : int, default = None
    Maximum number of iterations to perform. 
  tol : float
    convergence tolerance for early-stopping the iteration. 
  orth : int
    maximum number of Lanczos vectors to orthogonalize vectors against.
  sparse_mat : bool, default = False 
    Whether to collect the diagonal and off-diagonal terms into a sparse matrix for output. 
  return_basis : bool, default = False
    Whether to return the last 'ncv' orthogonal basis / 'Lanczos' vectors. 

  Description
  -----------
  This function implements the Lanczos method, or as Lanczos called it, the _method of minimized iterations_. 

  """
  ## Basic parameter validation
  n: int = A.shape[0]
  k: int = A.shape[1] if max_steps is None else min(max_steps, A.shape[1])
  dt = dtype if dtype is not None else (A @ np.zeros(A.shape[1])).dtype
  assert k > 0, "Number of steps must be positive!"
  
  ## Determine number of projections + lanczos vectors
  orth: int = k if orth < 0 or orth > k else orth
  ncv: int = max(orth, 2) if not(return_basis) else k
  
  ## Generate the starting vector if none is specified
  if v0 is None:
    rng = np.random.default_rng(seed)
    v0: np.ndarray = rng.uniform(size=A.shape[1], low=-1.0, high=+1.0).astype(dt)
  else: 
    v0: np.ndarray = np.array(v0).astype(dt)
  assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."

  ## Allocate the tridiagonal elements + lanczos vectors in column-major storage
  alpha, beta = np.zeros(k+1, dtype=np.float32), np.zeros(k+1, dtype=np.float32)
  Q = np.zeros((n,ncv), dtype=np.float32, order='F')
  assert Q.flags['F_CONTIGUOUS'] and Q.flags['WRITEABLE'] and Q.flags['OWNDATA']
  
  ## Call the procedure
  _lanczos.lanczos(A, v0, k, tol, orth, alpha, beta, Q)
  
  ## Format the output(s)
  if sparse_mat:
    from scipy.sparse import spdiags
    T = spdiags(data=[beta, alpha, np.roll(beta,1)], diags=(-1,0,+1), m=k, n=k)
    return T if not return_basis else (T, Q)
  else:
    a, b = alpha[:k], beta[1:k]
    return (a,b) if not return_basis else ((a,b), Q)


    