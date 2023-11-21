import primate
import numpy as np 
from scipy.sparse import csc_array
from scipy.sparse import spdiags
# T = spdiags(data=[beta, alpha, np.roll(beta,1)], diags=(-1,0,+1), m=n, n=n)

def lanczos_paige2(A, v0: np.ndarray, k: int, tol: float = 1e-8, ncv: int = 2):
  """Performs the k-step Lanczos tridiagonalization to a symmetric matrix A.
  
  The parameters k 

  Parameters: 
    A = Any type that implements the __matmul__ method, such as an ndarray, an sparray, or a LinearOperator.
    v = Starting vector for iteration.
    k = Maximum number of Lanczos steps to perform. 
    tol = Residual error criterion for early stopping. 
    ncv = Number of Lanczos vectors to orthogonalize against.
  
  Returns: 
    tuple (alpha, beta) representing the diagonal and subdiagonal, respectively, of the tridiagonal matrix T = Q^T A Q.
  """
  import _lanczos
  assert k <= A.shape[0], "Can perform at most k = n iterations"
  n, ncv = A.shape[0], max(2, ncv)
  alpha, beta = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
  Q = np.zeros(shape=(n, ncv), dtype=np.float32, order='F')
  Q[:,0] = v / np.linalg.norm(v)   
  for j in range(k):
    p,c,n = (j-1) % ncv, j % ncv, (j+1) % ncv
    Q[:,n] = A @ Q[:,c] - beta[j] * Q[:,p]
    alpha[j] = np.dot(Q[:,c], Q[:,n])
    Q[:,n] -= alpha[j] * Q[:,c]
    #print(Q)
    if ncv > 2: 
      _lanczos.orth_vector(Q[:,n], Q, c, ncv - 1, True) # orthogonalize 
    beta[j+1] = np.linalg.norm(Q[:,n])
    Q[:,n] /= beta[j+1] 
  return alpha, beta

    # if np.isclose(beta[j+1], tol):
    #   break
# // From Lanczos w/ Partial re-orthogonalization paper 
# def lanczos_paige2(A, v: np.ndarray, k: int, tol: float = 1e-8):
#   assert k <= A.shape[0], "Can perform at most k = n iterations"
#   n = A.shape[0]
#   alpha, beta = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
#   V = np.zeros(shape=(n, 2), dtype=np.float32)
#   m = V.shape[1]
#   V[:,0] = v
#   beta[0] = np.linalg.norm(v)
#   for j in range(k):
#     q = V[:,j % m] / beta[j]
#     u = A @ q - beta[j] * V[:,(j-1) % m]
#     alpha[j] = np.dot(u, q)    
#     V[:,(j+1) % m] = u - alpha[j] * q
#     beta[j+1] = np.linalg.norm(V[:,(j+1) % m])
#     if np.isclose(beta[j+1], tol):
#       break
#     print(V)
#     # print(f"{j % m} {} {}")
#   return alpha, beta

## Conclusion: Paiges A1 variant + surprisingly base variant show effectively no difference
## and are the best in terms of L2-approximation
def test_lanczos_eigen():
  from primate.diagonalize import _lanczos, lanczos
  np.random.seed(1234)

  winners = np.zeros(4)
  obj_loss = np.zeros(4)
  for i in range(1500):
    n = 20
    np.random.seed(1234+i) # 671 
    A = np.random.uniform(size=(n, n)).astype(np.float32)
    A = A @ A.T
    v = np.random.normal(size=n).astype(np.float32)
    true_ew = np.linalg.eigvalsh(A)
    A_sparse = csc_array(A)

    ## Test #1: Establish baseline
    a1, b1 = lanczos_base(A_sparse, v, k=n, tol=0.0)
    
    ## Test #2: Use Paige's A1 variant
    # a2, b2 = lanczos(A, v, n, 0.0, 0)
    a2, b2 = lanczos_paige(A_sparse, v, n, 0.0)

    ## Test #3: Use variation of paige's that re-orthogonalizes
    a3, b3 = lanczos_paige2(A_sparse, v, n, 0.0, 2)

    ## Test #4: Custom 
    a4, b4 = np.zeros(n+1, dtype=np.float32), np.zeros(n+1, dtype=np.float32)
    _lanczos.lanczos(A_sparse, v, n, 0.0, 5, a4, b4)
    assert not(np.any(np.isnan(a4)))

    from scipy.linalg import eigvalsh_tridiagonal
    ew1 = eigvalsh_tridiagonal(a1[:-1], b1[1:-1])
    ew2 = eigvalsh_tridiagonal(a2[:-1], b2[1:-1])
    ew3 = eigvalsh_tridiagonal(a3[:-1], b3[1:-1])
    ew4 = eigvalsh_tridiagonal(a4[:-1], b4[1:-1])

    d1 = np.linalg.norm(ew1 - true_ew)
    d2 = np.linalg.norm(ew2 - true_ew)
    d3 = np.linalg.norm(ew3 - true_ew)
    d4 = np.linalg.norm(ew4 - true_ew)
    min_dist = np.min([d1,d2,d3,d4])
    winners[np.isclose(np.abs(np.array([d1,d2,d3,d4]) - min_dist), 0.0)] += 1
    obj_loss += np.array([d1,d2,d3,d4])
    #print(f"1: {d1}, 2: {d2}, 3: {d3}, 4: {d4}, winner: {winners[-1]}")
  print(obj_loss / 1500)

def test_matrix_function():

  pass

# def test_benchmark():


def test_orthogonalize_vector():
  import primate
  import numpy as np 
  from primate.diagonalize import _lanczos
  def gen_U():
    np.random.seed(1234)
    U = np.random.uniform(size=(6,5)).astype(np.float32, order='F')
    U @= np.diag(np.reciprocal(np.linalg.norm(U, axis=0)))
    assert U.flags['F_CONTIGUOUS'] and U.flags['WRITEABLE'] and U.flags['OWNDATA']
    assert np.allclose(np.linalg.norm(U, axis=0), 1.0)
    return U
  
  ## Ensure the idea of orthogonalization by projection works
  U = gen_U()
  _lanczos.orth_vector(U[:,0], U, 1, 1, False)
  assert np.abs(np.dot(U[:,0], U[:,1])) <= 1e-6 

  ## Ensure last vector is always numerically orthogonal
  U = gen_U()
  _lanczos.orth_vector(U[:,0], U, 1, 2, False)
  assert np.abs(np.dot(U[:,0], U[:,2])) <= 1e-6 

  ## Ensure the degree of orthogonality improves each iteration with cycling 
  U, orth_factor = gen_U(), [0]*U.shape[1]
  for i in range(U.shape[1]-1):
    orth_factor[i] = np.sum(np.abs(U.T @ U))
    _lanczos.orth_vector(U[:,i], U, i+1, U.shape[1]-1, False)
    U[:,i] /= np.linalg.norm(U[:,i])
    new_orthogonality = np.sum(np.abs(U.T @ U))
    assert new_orthogonality < orth_factor[i]

  ## Ensure we basically have the identity 
  assert np.isclose(new_orthogonality, U.shape[1], atol=1e-5)
  assert np.max(np.abs(np.eye(U.shape[1]) - U.T @ U)) <= 1e-6

  # const auto u_norm = U.col(i).squaredNorm();
  # const auto proj_len = v.dot(U.col(i));
  # // std::cout << "i: " << i << ", u norm: " << u_norm << ", proj len: " << proj_len << ", tol: " << tol << std::endl; 
  # if (std::min(std::abs(proj_len), u_norm) > tol){
  #   v -= (proj_len / u_norm) * U.col(i);

  u, v = U[:,0], U[:,1]
  np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v))


def test_mgs():
  import primate
  import numpy as np 
  from primate.diagonalize import _lanczos
  np.random.seed(1234)
  U = np.random.uniform(size=(6,5)).astype(np.float32, order='F')
  assert U.flags['F_CONTIGUOUS'] and U.flags['WRITEABLE'] and U.flags['OWNDATA']
  _lanczos.mgs(U, 0)
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
