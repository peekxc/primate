import numpy as np 
from typing import *
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator
import _diagonalize

def lanczos(A: Union[LinearOperator, sparray], v0: Optional[np.ndarray] = None, max_steps: int = None, tol: float = 1e-8, orth: int = 0, sparse_mat: bool = False):
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
    convergence tolerance. 
  orth : int
    maximum number of Lanczos vectors to orthogonalize vectors against.
  sparse_mat : bool, default = False 
    Whether to collect the diagonal and off-diagonal terms into a sparse matrix for output. 

  Description
  -----------
  This function implements the Lanczos method, or as he called it, the method of minimized iterations. 

  """
  v0 = np.random.uniform(size=A.shape[1], low=-1.0, high=+1.0) if v0 is None else np.array(v0)
  assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."
  n = A.shape[0]
  max_steps = A.shape[1] if max_steps is None else min(max_steps, A.shape[1])
  assert max_steps > 0, "Number of steps must be positive!"
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  _diagonalize.lanczos_tridiagonalize(A, v0, max_steps, tol, orth, alpha, beta)
  if sparse_mat:
    from scipy.sparse import spdiags
    T = spdiags(data=[beta, alpha, np.roll(beta,1)], diags=(-1,0,+1), m=n, n=n)
    return T
  else:
    return alpha, beta


# import numpy as np 
# def eigh_tridiagonal(alphas: numpy.ndarray, betas: np.ndarray, V: np.ndarray):
#   _diagonalize.eigh_tridiagonal(alphas, betas, V)
#   return V
