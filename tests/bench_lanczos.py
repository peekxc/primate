import primate
import numpy as np 
from scipy.sparse import csc_array
from scipy.sparse import spdiags
  # T = spdiags(data=[beta, alpha, np.roll(beta,1)], diags=(-1,0,+1), m=n, n=n)
  

## Very basic Lanczos iteration
def lanczos_base(A, v: np.ndarray, k: int, tol: float = 1e-8):
  assert k <= A.shape[0], "Can perform at most k = n iterations"
  n = A.shape[0]
  alpha, beta = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
  qp, qc = np.zeros(n, dtype=np.float32), (v / np.linalg.norm(v)).astype(np.float32)
  for i in range(k):
    qn = A @ qc - beta[i] * qp
    alpha[i] = np.dot(qn, qc)
    qn -= alpha[i] * qc
    beta[i+1] = np.linalg.norm(qn)
    if np.isclose(beta[i+1], tol):
      break
    qn = qn / beta[i+1]
    qp, qc = qc, qn
  return alpha, beta

def lanczos_paige(A, v: np.ndarray, k: int, tol: float = 1e-8):
  assert k <= A.shape[0], "Can perform at most k = n iterations"
  n = A.shape[0]
  alpha, beta = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
  V = np.zeros(shape=(n, 2), dtype=np.float32)
  V[:,0] = v / np.linalg.norm(v)  # v
  V[:,1] = A @ V[:,0]             # u
  for j in range(k):
    alpha[j] = np.dot(V[:,0], V[:,1])
    w = V[:,1] - alpha[j]*V[:,0]
    beta[j+1] = np.linalg.norm(w)
    if np.isclose(beta[j+1], tol):
      break
    vn = w / beta[j+1]
    V[:,1] = A @ vn - beta[j+1] * V[:,0]
    V[:,0] = vn
  return alpha, beta


## Conclusion: Paiges A1 variant + surprisingly base variant show effectively no difference
## and are the best in terms of L2-approximation
def test_lanczos_eigen():
  from primate.diagonalize import _lanczos, lanczos
  np.random.seed(1234)

  winners = np.zeros(4)
  obj_loss = np.zeros(4)
  for i in range(1500):
    n = 10
    A = np.random.uniform(size=(n, n)).astype(np.float32)
    A = A @ A.T
    v = np.random.normal(size=n)
    true_ew = np.linalg.eigvalsh(A)

    ## Test #1: establish baseline
    a1, b1 = lanczos_base(A, v, k=n, tol=0.0)
    
    ## Test #2: Use the imate implementation 
    a2, b2 = lanczos(A, v, n, 0.0, 0)

    ## Test #3: Try the custom
    a3, b3 = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    A_sparse = csc_array(A)
    _lanczos.lanczos(A_sparse, v, n, 0.0, 0, a3, b3)

    ## Test #4: Use paige's 'stable' A1 variant 
    a4, b4 = lanczos_paige(A, v, n, 0.0)

    from scipy.linalg import eigvalsh_tridiagonal
    ew1 = eigvalsh_tridiagonal(a1[:-1], b1[1:-1])
    ew2 = eigvalsh_tridiagonal(a2, b2[:-1])
    ew3 = eigvalsh_tridiagonal(a3, b3[:-1])
    ew4 = eigvalsh_tridiagonal(a4[:-1], b4[1:-1])

    d1 = np.linalg.norm(ew1 - true_ew)
    d2 = np.linalg.norm(ew2 - true_ew)
    d3 = np.linalg.norm(ew3 - true_ew)
    d4 = np.linalg.norm(ew4 - true_ew)
    min_dist = np.min([d1,d2,d3,d4])
    winners[np.isclose(np.abs(np.array([d1,d2,d3,d4]) - min_dist), 0.0)] += 1
    obj_loss += np.array([d1,d2,d3,d4])
    #print(f"1: {d1}, 2: {d2}, 3: {d3}, 4: {d4}, winner: {winners[-1]}")

  np.bincount(np.array(winners)-1)


  pass


def test_orthogonalize_eigen():
  import primate
  import numpy as np 
  from primate.diagonalize import _lanczos
  np.random.seed(1234)
  U = np.random.uniform(size=(6,5)).astype(np.float32, order='F')
  assert U.flags['F_CONTIGUOUS'] and U.flags['WRITEABLE'] and U.flags['OWNDATA']
  # _lanczos.
  # _lanczos.orthogonalize(U, U[:,0], [1,2,3])
  # _lanczos.mgs(U)
  # np.diag(U.T @ U)
  
  np.random.seed(1234)
  U = np.random.uniform(size=(6,5)).astype(np.float32, order='F')
  print(U)
  _lanczos.mgs(U, 0)
  print(U)
  QI = U.T @ U
  assert np.max(np.abs(QI - np.diag(QI.diagonal()))) <= 1e-6
  assert np.max(np.abs(1.0 - QI.diagonal())) <= 1e-6

  

  np.random.seed(1234)
  U = np.random.uniform(size=(6,5)).astype(np.float32, order='F')
  v = U[:,0].copy()
  _lanczos.orth_vector(v, U, 1, 4, False)
  print(v)


  proj = lambda v, u: np.dot(u,v)/np.dot(u,u) * u # project v onto u 
  np.random.seed(1234)
  U = np.random.uniform(size=(6,5)).astype(np.float32, order='F')
  uk = U[:,0].copy()
  uk -= proj(uk, U[:,1])
  uk -= proj(uk, U[:,2])
  uk -= proj(uk, U[:,3])
  uk -= proj(uk, U[:,4])
  # uk -= proj(uk, U[:,0])
  uk = uk / np.linalg.norm(uk)
  print(uk)


  np.max(np.triu(np.abs(U.T @ U), 1))
  np.linalg.norm(U, axis=0)

  # np.linalg.norm(U, axis=1)
  pass
