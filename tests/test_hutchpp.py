import numpy as np 
from primate.random import symmetric
from primate.trace import hutch


def test():
  np.random.seed(1234)
  ew = np.sort(np.random.uniform(size=100, low=0.0, high=1.0))
  decay_spectrum = lambda beta: np.exp(-beta* ew)
  
  from primate.random import symmetric
  A = symmetric(n=100, ew=decay_spectrum(1e+1))

  ## Let's verify this normalization phenonenon
  d, b = A.shape[0], 20
  W = np.random.normal(size=(d, b))
  Q, R = np.linalg.qr(A @ W)

  ## Verify split works
  tr_true = np.trace(A)
  lr_true = np.trace(Q.T @ A @ Q)
  rs_true = np.trace((np.eye(d) - Q @ Q.T) @ A @ (np.eye(d) - Q @ Q.T))
  print(f"True: {tr_true:.5f}, Split: {lr_true + rs_true:.5f} ({lr_true:.5f}) + ({rs_true:.5f})")
  assert np.isclose(tr_true, lr_true + rs_true)

  ## Project to the right range + normalize to unit 
  P = np.eye(d) - Q @ Q.T # identity operator on the complement of the column span of A
  Y = P @ W
  Y_norm = Y @ np.diag(np.reciprocal(np.linalg.norm(Y, axis = 0)))
  rs_est1 = (d - b) * np.mean([(y @ A @ y)/np.linalg.norm(y) for y in Y.T])
  rs_est2 = (d - b) * np.mean([y @ A @ y for y in Y_norm.T])
  rs_est3 = np.mean([(((d - b) / np.linalg.norm(y)) * y) @ A @ (((d - b) / np.linalg.norm(y)) * y) for y in Y.T]) / (d-b)
  print(f"Residual ests: (1) {rs_est1:.6f}, (2) {rs_est2:.6f}, (3) {rs_est3:6f} <=> (true) {rs_true:.6f}")
  print(f"Errors: (1) {np.abs(rs_est1 - rs_true):.6f}, (2) {np.abs(rs_est2 - rs_true):.6f}, (3) {np.abs(rs_est3 - rs_true):6f}")
  ## Looks like (2) and (3) are almost always superior over (1), but are always identical as well
  assert np.isclose(rs_est2, rs_est3)

  ## See if unbiased 
  W = np.random.normal(size=(d, b*100))
  Y = P @ W
  Y_norm = Y @ np.diag(np.reciprocal(np.linalg.norm(Y, axis = 0)))
  rs_est = (d - b) * np.mean([y @ A @ y for y in Y_norm.T])
  print(f"Res true: {rs_true:.5f}, High acc. approx: {rs_est:.5f}")

  

  ## XTrace normalization
  # np.mean([y @ A @ y for y in Y_norm.T])
  # np.sqrt(d - (b - 1)) * 



  # ## Unnormalized Girard-Hutchinson 
  # def quad_form(A):
  #   v = np.random.normal(size=A.shape[0], loc=0, scale=1)
  #   v /= np.linalg.norm(v)
  #   return v @ A @ v

  # ## Simple girard Hutchinson 
  # print(f"GH: {d * np.mean(np.array([quad_form(A) for _ in range(2500)])):.5f}, True: {tr_true:.5f}")



  # residuals = np.array([quad_form(A) for _ in range(2500)])
  # res_est = np.mean(residuals)  
  # # ((d-b)/len(residuals)) * np.sum(residuals)
  # print(f"error: {tr_true - (lr_est + res_est):.5f}")



   
  assert True



def test_ada_krylov_aware_trace():
  from primate.random import symmetric
  from primate.diagonalize import lanczos
  from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal

  np.random.seed(1234)
  ew = np.sort(np.random.uniform(size=100, low=0.0, high=1.0))
  decay_spectrum = lambda beta: np.exp(-beta* ew)
  A = symmetric(n=100, ew=decay_spectrum(1.0))
  print(f"Tr(A) = {np.trace(A):.5f}")

  eps, delta = 0.1, 0.75
  n = 10
  d, b = A.shape[0], 1 
  W = np.random.normal(size=(d, b))
  q = 20 
  (av,bv), Q = lanczos(A, v0 = W, deg=q+n, return_basis=True)
  QP = Q[:,:q]

  rw = eigvalsh_tridiagonal(av,bv)
  t_defl = np.sum(rw)

  t_rem, t_fro = 0.0, 0.0
  m, k = np.inf, 0
  while k < 100: #m > k:
    k += 1
    p = np.random.normal(size=(d, 1))
    y = p - Q @ (Q.T @ p)
    y /= np.linalg.norm(p)
    av, bv = lanczos(A, v0=y, deg=n)
    rw, rv = eigh_tridiagonal(av,bv)
    t_rem += sum(rw * rv[0,:]**2) * np.linalg.norm(y)**2 # z @ (rv @ np.diag(rw) @ rv.T) @ z
    t_fro0 = np.linalg.norm((rv @ np.diag(rw) @ rv.T)[:,0])
    t_fro += t_fro0**2 * np.linalg.norm(y)**2
    print(f"T est: {t_defl + (1/k) * t_rem:.5f} (k = {k})")

  t_defl + (1/k) * t_rem

  pass 
  

# class ComplementProjector(LinearOperator):
#   def __init__(Q):
#     self.Q = Q
#     self.shape = (Q.shape[0], Q.shape[0])
#     self.dtype = Q.dtype
  
#   def _matvec(x):
#     x = x.reshape(-1)
#     y = self.Q @ (self.Q.T @ x)
#     x -= y