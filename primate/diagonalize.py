import numpy as np 
from typing import *
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator
from primate import _diagonalize

def lanczos(A: Union[LinearOperator, sparray], v0: Optional[np.ndarray] = None, tol: float = 1e-8, orth: int = 0, sparse_mat: bool = False):
  n = A.shape[0]
  v0 = np.random.uniform(size=A.shape[1], low=-1.0, high=+1.0) if v0 is None else v0 
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  _diagonalize.lanczos_tridiagonalize(A, v0, tol, orth, alpha, beta)
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
