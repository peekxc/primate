import numpy as np 
from numpy.linalg import eigh
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')


def test_shift_invert():
  from primate.random import symmetric
  from primate.operator import ShiftedInvOp
  np.random.seed(1234)
  ew = np.linspace(0, 1, 100) #np.random.uniform(size=100, low=0, high=1)
  ew[np.argsort(ew)[:15]] = 1e-14
  A = symmetric(n=len(ew), ew=ew, pd=False)


  pos_ew = np.sort(ew)[15:]
  width = np.min(np.diff(pos_ew)) / 4 ## the constant matters here; 
  A_OpInv = ShiftedInvOp(A)
  for i, sigma in enumerate(pos_ew):
    A_OpInv.sigma = sigma + width
    ew_right = eigsh(A_OpInv, which='LM', k=1, sigma=A_OpInv.sigma, OPinv=A_OpInv, return_eigenvectors=False)
    A_OpInv.sigma = sigma - width
    ew_left = eigsh(A_OpInv, which='LM', k=1, sigma=A_OpInv.sigma, OPinv=A_OpInv, return_eigenvectors=False)
    assert np.isclose(ew_right, sigma) and np.isclose(ew_left, sigma), "Shift-invert failed"
    print(i)


def test_spectral_gap():
  from primate.random import symmetric
  from primate.operator import ShiftedInvOp

  np.random.seed(1234)
  ew = np.random.uniform(size=300, low=0, high=1)
  ew[np.argsort(ew)[:15]] = 1e-14
  A = symmetric(n=len(ew), ew=ew, pd=False)
  gap = np.min(ew[ew > 1e-14])

  ## First step: try shift-invert with sigma = 0.0
  from scipy.optimize import golden, minimize_scalar
  tol = np.max(ew) * np.max(A.shape) * np.finfo(A.dtype).eps
  A_OpInv = ShiftedInvOp(A, sigma=0.0)
  def est_gap(sigma: float):
    A_OpInv.sigma = sigma
    gap_est = eigsh(A_OpInv, which='LM', k=1, sigma=A_OpInv.sigma, OPinv=A_OpInv, return_eigenvectors=False)
    return gap_est if gap_est > tol else np.inf
  res = minimize_scalar(est_gap, bounds=(0.01*gap, 10*gap), method="bounded")
  res.fun

# from importlib_resources import files
# files('primate')

def challenge1():
  pass 
  from scipy.sparse import load_npz
  data_dir = primate.__file__[:(primate.__file__.find('primate') + 8)] + "data"
  A = load_npz(data_dir + '/challenge_1328_rank871.npz')
  # largest: 5.857450021586329
  # smallest: 0.0038922391321746437
  top_ew = np.take(eigsh(A, k=1, which='LM', return_eigenvectors=False), 0)
  low_ew = eigsh(A, which='SM', ncv = A.shape[0], k=A.shape[0] - 872, return_eigenvectors=False)
  gap = min(low_ew[~np.isclose(low_ew, 0.0)])

  from scipy.sparse.linalg import eigsh
  # v0 = np.random.normal(size=A.shape[0]) * top_ew
  # rank_deficient_tol = info['tolerance']
  from scipy.optimize import minimize, LinearConstraint
  from scipy.sparse import dia_matrix
  # unit_constraint = LinearConstraint(A=dia_matrix(np.eye(A.shape[0])), lb=0.99, ub=1.01)
  ii = 0
  def rayleigh_quotient(x, mu: float = 1.0):
    global ii
    x = x[:,np.newaxis] if x.ndim == 1 else x
    beta = (x.T @ x).item()
    rq = np.ravel((x.T @ A @ x) / beta).item()
    return rq + 100.0*(np.linalg.norm(x) - 1.0)**2

  def rayleigh_quotient(x, mu: float = 1.0):
    global ii
    x = x[:,np.newaxis] if x.ndim == 1 else x
    beta = (x.T @ x).item()
    rq = np.ravel((x.T @ A @ x) / beta).item()
    g = ((2.0 / np.linalg.norm(x)**2) * (A @ x - rq * x))
    grad = g.flatten()

    # dist_to_ev = np.linalg.norm((A @ x) - (rq * x))**2
    # rq += dist_to_ev
    # ii += 1 
    # if rq < rank_deficient_tol:
    #   rq += (mu / 2) * (rq - rank_deficient_tol)**2 
    #   grad += mu * np.abs(rq - rank_deficient_tol) * grad
    return (rq, grad)
  # np.linalg.norm((A @ res.x) - (res.fun * res.x))

  ## Hessian product for the Rayleigh quotient
  ## Can optimize further but only if nfev 
  def hessp(x, p): 
    ## Based on: https://math.stackexchange.com/questions/4797858/hessian-of-the-rayleigh-quotient-frac-langle-x-ax-rangle-langle-x-x-rangle
    # 2 * (1.0/beta) * (A - np.eye(A.shape[0]) - x @ g.T  - g @ x.T)
    x = x[:,np.newaxis] if x.ndim == 1 else x
    p = p[:,np.newaxis] if p.ndim == 1 else p
    beta = (x.T @ x).item()
    rq, grad = rayleigh_quotient(x)
    g = grad[:,np.newaxis]
    h = 2 * (1.0/beta) * A @ p - x @ (g.T @ p)  - g @ (x.T @ p)
    return h.flatten()

  # low_ew[~np.isclose(low_ew, 0.0)]

  # COBYLA, SLSQP and trust-constr
  # minimize(rayleigh_quotient, x0=v0, jac=True, method="Newton-CG")
  v0 = v0 / np.linalg.norm(v0)
  res = minimize(rayleigh_quotient, x0=v0, jac=False, method="L-BFGS-B")

  
  np.linalg.norm(res.x)
  
  res = minimize(rayleigh_quotient, x0=v0, jac=False, method="TNC")
  res = minimize(rayleigh_quotient, x0=v0, jac=True, method="Newton-CG")
  # minimize(rayleigh_quotient, x0=v0, jac=True, method="Newton-CG", hessp=hessp)

  rayleigh_quotient(res.x + np.random.uniform(size=len(res.x)))

  eigsh(A, v0=res.x, k=1, which='LM', sigma=res.fun, return_eigenvectors=True)

  
  hutch(A, fun="smoothstep", a=0, b=gap * ratio, maxiter=30, seed=i)



  # minimize(rayleigh_quotient, x0=v0, method="COBYLA", constraints=unit_constraint)

  eigsh(A, which='SM', k = 1, v0 = v0, return_eigenvectors=False, maxiter=A.shape[0])

  ## This seems to work 
  from primate.functional import estimate_spectral_radius
  from primate.diagonalize import lanczos
  ## => means lanczos up to degree 'deg' should yield rayleigh ritz value w/ rel error <= 0.01
  ## This uses the theory to do so!
  eps = 0.1
  info = estimate_spectral_radius(A, rtol=eps, full_output=True)
  deg = info['deg_bound']
  a, b = lanczos(A, deg=deg)
  rr_max = info['spectral_radius']
  # rr_max = eigvalsh_tridiagonal(a,b, select='i', select_range=[deg-1, deg-1])[0]
  assert np.abs((rr_max - top_ew) / top_ew) <= eps
  assert top_ew <= (rr_max / (1.0 - eps))

  ## Estimating the spectral gap is quite hard!
  gap = info['gap']
  lb, ub = 1e-2 * gap, 1e2 * gap
  est_gap = lambda sigma: eigsh(A, k = 1, sigma=sigma, which='LM', return_eigenvectors=False, maxiter=1, tol=np.inf)
  from scipy.sparse.linalg import eigsh
  [est_gap(sigma) for sigma in np.linspace(0.00001*lb, ub, 5)]

  from scipy.sparse.linalg import qmr, bicg, cg, cgs, gmres, lgmres, minres, gcrotmk,tfqmr

  AA = CountMatvec(A)
  AA.sigma = lb
  x = np.random.uniform(size=AA.shape[1])
  b = AA @ x



  # qmr(AA, b)
  cg(AA, b, maxiter=150)[0]
  np.linalg.norm(cg(AA, b, maxiter=150)[0] - x)
  np.linalg.norm(cgs(AA, b, maxiter=150)[0] - x)
  np.linalg.norm(gmres(AA, b, maxiter=10)[0] - x)
  np.linalg.norm(lgmres(AA, b, maxiter=10)[0] - x)
  np.linalg.norm(minres(AA, b, maxiter=10)[0] - x)
  np.linalg.norm(gcrotmk(AA, b, maxiter=10)[0] - x)
  np.linalg.norm(tfqmr(AA, b, maxiter=10)[0] - x)

  for sigma in np.linspace(0.00001*lb, ub, 5):
    AA = CountMatvec(A)
    res = eigsh(AA, k = 1, sigma=sigma, which='LM', return_eigenvectors=False, maxiter=3, tol=0.0)
    print(res)
    print(AA.ii)
    
  from primate.random import symmetric
  ew = np.random.uniform(size=30, low=0, high=1)
  A = symmetric(n=30, ew=ew, pd=False)
  B = CountMatvec(A)
  B.sigma = 2 * np.min(ew)
  b = np.random.uniform(size=B.shape[1])
  

  ## This strategy seems to work
  class ShiftedOp(LinearOperator):
    def __init__(self, A, sigma = 0.0):
      self.A = A
      self.dtype = A.dtype 
      self.shape = A.shape
      self.sigma = sigma
    def _matvec(self, x):
      x = x[:,np.newaxis] if x.ndim == 1 else x
      return self.A @ x - self.sigma * x

  class ShiftedInvOp(LinearOperator):
    def __init__(self, A_shift):
      self.A_shift = A_shift
      self.dtype = A_shift.dtype 
      self.shape = A_shift.shape
      self.ii = 0
    def _matvec(self, x):
      self.ii += 1
      return cg(self.A_shift, x)[0]
  
  ew = np.sort(ew) 
  delta = np.min(np.diff(ew))
  for sigma in ew*0.80:
    op_inv = ShiftedInvOp(ShiftedOp(A, sigma=sigma))
    sh_ew = eigsh(A, which='LM', k=1, sigma=sigma, OPinv=op_inv, return_eigenvectors=False)
    assert np.argmin(np.abs(sigma - ew)) == np.argmin(np.abs(sh_ew - ew)), "Shifted guess did not converge to correct eigenvalue"
    assert np.min(np.abs(sh_ew - ew)) < 1e-6, "Shifted eigenvalue did not converge within tolerance"

  # cg(ShiftedOp(A, sigma=0.01), x)[0]


  eigsh(A, which='LM', k=5, sigma=0.0, return_eigenvectors=False)

  np.sort(ew)

  op_inv.ii

  # ShiftedInvOp(ShiftedOp(A, sigma=0.01)) @ x


  def inv_op_shifted(A, sigma):
    ShiftedOp
    return scipy.sparse.linalg.spsolve(B, v)

  eigsh(A, k=1, which='LM', sigma=0.09, OPinv=OP, return_eigenvectors=False)
  
  OP @ x

  np.linalg.norm(x) - np.linalg.norm(B @ x)
  

  # from scipy.sparse.linalg import eigsh
  from scipy.sparse.linalg._eigen.arpack import _SymmetricArpackParams
  # eigsh(A, k=1, which='LM', sigma=B.sigma, OPinv=B)
  #  params = _SymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
  #                                   M_matvec, Minv_matvec, sigma,
  #                                   ncv, v0, maxiter, which, tol)
  #   with _ARPACK_LOCK:
  #       while not params.converged:
  #           params.iterate()
    #  return IterOpInv(_aslinearoperator_with_dtype(A),
    #                         #  M, sigma, tol=tol).matvec
  # from scipy.sparse.linalg import get_OPinv_matvec
  #               Minv_matvec = get_OPinv_matvec(A, M, sigma,
  #                                              hermitian=True, tol=tol)  


  from primate.trace import hutch
  errors = []
  for ratio in [2, 1.5, 1.1, 1, 0.9, 0.5, 0.25, 1/8]:
    nr = np.array([hutch(A, fun="smoothstep", a=0, b=gap * ratio, maxiter=30, seed=i) for i in range(250)])
    errors.append(np.mean(np.abs(nr - 871)))

  ## Indeed, whereas error is relatively small (~3) for gap * 2 and gap, the computation is highly unstable 
  ## if the gap is set too low! On the other hand, the computation introduces bias if gap is too large!
  ## In contrast, the error remains stable if we remain above the gap 
  ## So, ideally, we want to use an upper-bound on the gap
  # from scipy.optimize


import numpy as np
from scipy.optimize import minimize

## Rayleigh quotient with Lagrange multiplier
# def rayleigh_quotient_lagrange(v, A, mu: float = 1.0):
#   u = v / np.linalg.norm(v)  # Normalize u
#   u = u[:,np.newaxis] if u.ndim == 1 else u
#   rayleigh = u.T @ A @ u
#   return rayleigh # - mu * (np.dot(u.T, u) - 1)
def rayleigh_quotient(x, A, mu: float = 1.0, eps: float = 1.0):
    x = x[:, np.newaxis] if x.ndim == 1 else x
    x_norm = np.linalg.norm(x)
    
    # Compute the Rayleigh quotient
    rq = (x.T @ A @ x).item() / (x.T @ x).item()
    
    # Compute the gradient of the Rayleigh quotient
    g_rq = (2.0 / x_norm**2) * (A @ x - rq * x)
    
    # Add the quadratic penalty term to the objective
    penalty = (1.0 / (2.0 * mu)) * (x_norm - 1.0)**2
    
    # Compute the gradient of the penalty term
    g_penalty = (1.0 / mu) * (x_norm - 1.0) * (x / x_norm)
    
    # Quadratic penalty for the inequality constraint
    penalty_ineq = (1.0 / (2.0 * eps)) * (1.0 + max(0, gap - rq))**2
    g_penalty_ineq = np.zeros_like(x)
    if rq < gap:
      g_penalty_ineq = -(1.0 / eps) * (gap - rq) * g_rq

    # Combine the gradients
    grad = g_rq + g_penalty + g_penalty_ineq
    grad = grad.flatten()
    
    return rq + penalty + penalty_ineq, grad

# cons = {'type': 'eq', 'fun': lambda v: np.linalg.norm(v) - 1}
# v0 = v0 / np.linalg.norm(v0)
for eps in np.linspace(0.001, 1, 30):
  res = minimize(rayleigh_quotient, v0.flatten(), args=(A, 1.0, eps), jac=True, method="TNC")
  # print(f"Result: {res}")
  print(f"(eps = {eps:.4f}), Norm of x: {np.linalg.norm(res.x):.5f}, Ax scale: {np.linalg.norm(A @ res.x):.8f}, dist to gap: {np.abs(gap-np.linalg.norm(A @ res.x))}")


## Idea: use shift-invert over varying sigma form [0 -> 10 * tol]
from scipy.sparse.linalg import eigsh, aslinearoperator, LinearOperator
class CountMatvec(LinearOperator):
  def __init__(self, A, sigma: float = 0.0):
    self.A = A
    self.ii = 0
    self.shape = A.shape
    self.dtype = A.dtype
    self.sigma = sigma
  def reset(self):
    self.ii = 0
  # def _matvec(self, v):
  #   self.ii += 1
  #   return self.A @ v
  def _matmat(self, v):
    v = v[:,np.newaxis] if v.ndim == 1 else v
    self.ii += v.shape[1]
    return self.A @ v - self.sigma * v 

M = CountMatvec(A)

eigsh(M, k = 1, sigma=1e-6, which='LM', return_eigenvectors=False, maxiter=10)
eigsh(A, k = 1, sigma=1e-5, which='LM', return_eigenvectors=False)
eigsh(A, k = 1, sigma=1e-4, which='LM', return_eigenvectors=False)
eigsh(A, k = 1, sigma=1e-3, which='LM', return_eigenvectors=False)
eigsh(A, k = 1, sigma=0.005, which='LM', return_eigenvectors=False)
eigsh(A, k = 1, sigma=1e-2, which='LM', return_eigenvectors=False)

res = minimize(rayleigh_quotient_lagrange, v0, args=(A,10.0), method="L-BFGS-B")


minimize(rayleigh_quotient_lagrange, v0, args=(A,10.0), method="L-BFGS-B")


def rayleigh_quotient_grad_lagrange(v, A):
    n = len(v) // 2
    u, lmbda = v[:n], v[n]
    u = u / np.linalg.norm(u)  # Normalize u

    # Gradient with respect to u
    grad_u = 2 * (np.dot(A, u) - lmbda * u)

    # Gradient with respect to lambda
    grad_lmbda = 1 - np.dot(u.T, u)

    return np.concatenate([grad_u, [grad_lmbda]])

def smallest_eigen(A):
    n = A.shape[0]
    # Initial guess for the eigenvector and Lagrange multiplier
    x0 = np.random.rand(n + 1)
    
    # Optimize the Rayleigh quotient with Lagrange multiplier
    result = minimize(rayleigh_quotient_lagrange, x0, args=(A,), jac=rayleigh_quotient_grad_lagrange)

    # Extract and normalize the resulting eigenvector
    eigenvector = result.x[:n]
    eigenvector = eigenvector / np.linalg.norm(eigenvector)
    # Compute the corresponding eigenvalue
    eigenvalue = rayleigh_quotient_lagrange(result.x, A) + result.x[n]  # Remove Lagrange multiplier contribution
    
    return eigenvalue, eigenvector