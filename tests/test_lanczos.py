import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 
from primate.random import symmetric

## Add the test directory to the sys path 
import sys
import primate
rel_dir = primate.__file__[:(primate.__file__.find('primate') + 7)]
sys.path.insert(0, rel_dir + '/tests')

def test_basic():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 100
  A_sparse = csc_array(symmetric(n))
  v0 = np.random.uniform(size=n)
  (a,b), Q = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=n-1, return_basis = True)
  assert np.abs(max(eigh_tridiagonal(a,b, eigvals_only=True)) - max(eigsh(A_sparse)[0])) < 1e-4

  from primate.diagonalize import _lanczos
  true_ew = eigh_tridiagonal(a,b, eigvals_only=True)
  test_rw = _lanczos.ritz_values(a, np.append(0,b), n)
  assert np.allclose(np.sort(test_rw), np.sort(true_ew))

# from primate.functional import estimate_spectral_gap
# estimate_spectral_gap(A)

def test_matvec():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 10
  A_sparse = csc_array(symmetric(n))
  v0 = np.random.uniform(size=n)
  (a,b), Q = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=0, return_basis = True) 
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
  y = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))
  z = A_sparse @ v0
  assert np.linalg.norm(y - z) < 1e-4

def test_accuracy():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = symmetric(n).astype(np.float64)
  alpha, beta = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  tol = np.zeros(30)
  for i in range(30):
    v0 = np.random.uniform(size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-8, orth=n-1)
    ew_true = np.sort(eigsh(A, k=n-1, return_eigenvectors=False))
    ew_test = np.sort(eigh_tridiagonal(alpha, beta, eigvals_only=True))
    tol[i] = np.max(np.abs(ew_test[1:] - ew_true))
  assert np.all(tol < 1e-12)

def test_high_degree():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 30
  A = symmetric(n)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  for k in range(2, 30):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=k, orth=0)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))

def test_degree_unbiased():
  from primate.diagonalize import lanczos
  from scipy.linalg import eigvalsh_tridiagonal
  np.random.seed(1234)
  n = 150
  A = symmetric(n)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  ew = np.linalg.eigvalsh(A)
  for k in range(5, 150, 5):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=k, orth=15)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))
    rr = eigvalsh_tridiagonal(alpha, beta)
    assert np.allclose(rr.mean(), ew.mean(), atol=0.30)

def test_toeplitz():
  from scipy.linalg import toeplitz
  from primate.diagonalize import lanczos
  from scipy.linalg import eigvalsh_tridiagonal
  np.random.seed(1234)
  n = 150
  c = np.random.uniform(size=n, low=0, high=1.0)
  A = toeplitz(c)
  alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
  
  ## In general not guaranteed, but with full re-orthogonalization it's likely (and deterministic w/ fixed seed)
  ew = np.linalg.eigvalsh(A)
  for k in range(25):
    v0 = np.random.choice([-1.0, +1.0], size=A.shape[1])
    alpha, beta = lanczos(A, v0=v0, rtol=1e-7, deg=150, orth=15)
    assert np.all(~np.isnan(alpha)) and np.all(~np.isnan(beta))
    rr = eigvalsh_tridiagonal(alpha, beta)
    assert np.allclose(rr.mean(), ew.mean(), atol=np.ptp(ew)*0.05)

## TODO: go back and re-test fttr on toeplitz
# def test_symmetric():
#   assert np.allclose(ew_sym, np.flip(-ew_sym))



def test_quadrature():
  from primate.diagonalize import lanczos, _lanczos
  from sanity import lanczos_quadrature
  np.random.seed(1234)
  n = 10
  A = csc_array(symmetric(n))
  v0 = np.random.uniform(size=A.shape[1])

  ## Test the near-equivalence of the weights and nodes
  alpha, beta = lanczos(A, v0, deg=n, orth=n)  
  nw_test = _lanczos.quadrature(alpha, np.append(0, beta), n, 0)
  nw_true = lanczos_quadrature(A, v=v0, k=n, orth=n)
  assert np.allclose(nw_true[0], nw_test[:,0], atol=1e-6)
  assert np.allclose(nw_true[1], nw_test[:,1], atol=1e-6)

def test_fttr_basic():
  from scipy.sparse import spdiags
  alpha = np.array([1,1,1])
  beta = np.array([1,1,0])
  T = spdiags(data=[beta, alpha, np.roll(beta, 1)], diags=(-1, 0, +1), m=3, n=3).todense()
  ew, ev = np.linalg.eigh(T)

  a = alpha 
  b = np.append(0, beta[:-1])
  mu_0 = np.sum(np.abs(ew))
  p0 = lambda x: 1 / np.sqrt(mu_0)
  p1 = lambda x: (x - a[0])*p0(x) / b[1]
  p2 = lambda x: ((x - a[1])*p1(x) - b[1] * p0(x)) / b[2]
  p = lambda x: np.array([p0(x), p1(x), p2(x)])
  weights_fttr = np.reciprocal([np.sum(p(lam) ** 2) for lam in ew])

  ## Forward three-term recurrence relation (fttr)
  assert np.allclose(weights_fttr, mu_0 * np.ravel(ev[0,:])**2)

def test_fttr_2():
  from sanity import orth_poly
  from scipy.sparse import spdiags
  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  from scipy.linalg import toeplitz
  n = 5
  alpha = np.random.uniform(size=n, low=0, high=1)
  beta = np.append(np.random.uniform(size=n-1, low=0, high=1), 0)
  T = spdiags(data=[beta, alpha, np.roll(beta, 1)], diags=(-1, 0, +1), m=n, n=n).todense()
  ew, ev = np.linalg.eigh(T)

  ## Deduced as the FTTR algorothm from Theorem 1 of "Computing Gaussian quadrature rules with high relative accuracy"
  a = alpha 
  b = np.append(0, beta[:-1]) # first must be zero
  mu_0 = np.sum(np.abs(ew))
  p = lambda x: np.array([orth_poly(x, i, mu_0, a, b) for i in range(n)])
  
  weights_fttr = np.reciprocal([np.sum(p(lam) ** 2) for lam in ew])
  assert np.allclose(weights_fttr, mu_0 * np.ravel(ev[0,:])**2)

def test_fttr3():
  from scipy.linalg import toeplitz
  from scipy.sparse import spdiags
  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  n = 8
  A = toeplitz([0,1,2,3,4,5,6,7]).astype(np.float64) # symmetric(n)
  v0 = np.random.uniform(size=A.shape[1])
  alpha, beta = lanczos(A, v0=v0, deg=n, orth=n-1)
  a, b = alpha, np.append([0], beta)
  T = spdiags(data=[np.roll(b,-1), a, b], diags=(-1, 0, +1), m=n, n=n).todense()
  # T = np.array([[4, 3, 0], [3, 1, 1], [0, 1, -1]])
  ew, ev = np.linalg.eigh(T)
  a, b = np.diag(T, 0), np.append([0], np.diag(T, 1))
  mu_0 = np.sum(np.abs(ew))

  # echo = lambda i, x, a1, z1, b1, z0, b2: print(f"{i}: (({x} - {a1}) * {z1} - {b1} * {z0}) / {b2}")
  # def p(x, mu_0, a, b, verbose: bool = True):
  #   z0 = 1 / np.sqrt(mu_0)
  #   z1 = (x - a[0]) * z0 / b[1]
  #   if verbose:
  #     echo(1, x, a[0], z0, b[1], 0, 0)
  #   z = [z0,z1]
  #   for i in range(2, len(a)):
  #     zi = ((x - a[i-1]) * z[i-1] - b[i-1] * z[i-2]) / b[i]
  #     z += [zi]
  #     if verbose: 
  #       echo(i, x, a[i-1], z[i-1], b[i-1], z[i-2], b[i])
  #   return np.array(z)

  # from primate.ortho import _orthogonalize
  # fttr_weights_base = _orthogonalize.fttr(ew, a, b)
  fttr_weights_run = _lanczos.quadrature(a,b,len(a),1)[:,1]
  fttr_weights_true = np.ravel(ev[0,:])**2
  assert np.allclose(fttr_weights_run, fttr_weights_true)

def test_quadrature_methods():
  from primate.diagonalize import _lanczos
  n = 3
  a, b = [1,1,1], [0,1,1]
  nw_test1 = _lanczos.quadrature(a, b, n, 0)
  nw_test2 = _lanczos.quadrature(a, b, n, 1)
  assert np.allclose(nw_test1, nw_test2)

  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  n = 50 
  A = symmetric(n)
  v0 = np.random.uniform(size=A.shape[1])
  a, b = lanczos(A, v0=v0, deg=n)
  a, b = a, np.append([0], b)
  quad1 = np.sum(_lanczos.quadrature(a, b, n, 0).prod(axis=1))
  quad2 = np.sum(_lanczos.quadrature(a, b, n, 1).prod(axis=1))
  assert np.isclose(quad1, quad2, atol=quad1*0.05)

def test_tridiagonal():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 100
  A_sparse = csc_array(symmetric(n))
  v0 = np.random.uniform(size=n)
  M = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=5, sparse_mat=True).todense()
  a, b = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=5, sparse_mat=False)
  assert np.allclose(np.diag(M), a)
  assert np.allclose(np.diag(M, 1), b)

  # ## Gerschgorin disk check
  # radius = np.append([0], np.abs(b) + np.roll(np.abs(b), -1))
  # radius[0] = np.abs(b[0])
  # radius[-1] = np.abs(b[-1])
  # assert np.allclose((M - np.diag(a)).sum(axis=0), radius)
  # np.max(a + radius)
  # np.max(np.linalg.eigh(A_sparse.todense())[0])


## Testing Parlett's bounds on the spectral gap 
def test_rayleigh_approx():
  from primate.diagonalize import lanczos
  from scipy.linalg import eigh_tridiagonal
  from scipy.sparse.linalg import eigsh
  from primate.random import symmetric

  lb_works = []
  lb_error = []
  for i in range(50):
    ew_true = np.random.uniform(size=150, low=0.0, high=5.0)
    ew_true[ew_true <= .40] = 0.0
    A = symmetric(150, ew=ew_true, pd=False)
    a,b = lanczos(A, deg = 20)
    rr, rv = eigh_tridiagonal(a,b)
    tol = max(rr) * A.shape[0]  * np.finfo(A.dtype).eps
    rr = np.abs(rr)
    min_id = np.flatnonzero(rr >= tol)[np.argmin(rr[rr >= tol])]
    coeff = min([b[min_id-1], b[min_id], b[min_id+1]])
    ew_lb = rr[min_id] - coeff * np.abs(rv[-1,min_id])
    min_ew = min(ew_true[ew_true > tol])
    lb_works.append((rr[min_id] - 2*(coeff * np.abs(rv[-1,min_id]))) < min_ew)
    # print(f"LB Error, error est: {abs(min_ew - ew_lb):.5f}, {coeff * np.abs(rv[-1,min_id]):.5f}")
  assert np.sum(lb_works) > 45
  # ## Confirmed it does not work, unles algebraically smallest includes negative
  # min_rr - rv[:,min_id][-1] <= np.min(np.abs(ew_true))
  # eigsh(A, k=1, sigma = max(rr), which = 'SM', maxiter = 15, tol = 0.001, return_eigenvectors=False)

def test_rank_deficient():
  from primate.diagonalize import lanczos
  from primate.random import symmetric
  from scipy.linalg import eigh_tridiagonal
  ew = np.random.uniform(size=30, low=0.0, high=1.0)
  ew[:20] = 0.0
  A = symmetric(30, ew=ew)
  a,b = lanczos(A, deg=30, orth=10)
  assert np.all(~np.isnan(a)) and np.all(~np.isnan(b))
  rw, rv = eigh_tridiagonal(a,b)
  assert np.all(~np.isnan(rw)) and np.all(~np.isnan(np.ravel(rv)))

# def test_diagonal():
#   from primate.diagonalize import lanczos
#   from primate.random import symmetric
#   ew = np.random.uniform(size=30, low=0.0, high=1.0)
#   ew[:20] = 0.0
#   a,b = lanczos(np.diag(ew), deg=30)
#   rw, rv = eigh_tridiagonal(a,b)

#   from primate.trace import hutch
#   hutch(np.eye(10))
  

# def test_quadrature_toeplitz():
#   from primate.diagonalize import lanczos, _lanczos
#   from scipy.linalg import toeplitz
#   np.random.seed(1234)

#   A = toeplitz(np.arange(1,8)).astype(np.float32)
#   v0 = np.random.uniform(size=A.shape[1])
#   n = A.shape[1]
#   a, b = lanczos(A, deg=n)
#   a, b = a, np.append([0], b)
#   quad1 = np.sum(_lanczos.quadrature(a, b, n, 0).prod(axis=1))
#   quad2 = np.sum(_lanczos.quadrature(a, b, n, 1).prod(axis=1))
#   assert np.isclose(quad1, quad2, atol=np.abs(quad1)*0.50)

#   ## ground truth
#   ew, ev = np.linalg.eigh(A)
#   mu_0 = np.sum(np.abs(ew))
#   mu_0 * np.ravel(ev[0,:])**2
#   np.array([orth_poly(ew[0], i, mu_0, a, b)**2 for i in range(n)])
#   _lanczos.quadrature(a, b, n, 0)[:,1]
# np.linalg.norm([orth_poly(ew[0], i, mu_0, a, b) for i in range(n)])
# ## to mimick 
# arr = np.array([orth_poly(ew[0], i, mu_0, a, b)**2 for i in range(n)])

# n = 50 
# c = np.random.uniform(size=n, low=0, high=1)
# A = toeplitz(c)
# v0 = np.random.uniform(size=A.shape[1])
# a, b = lanczos(A, deg=n)
# a, b = a, np.append([0], b)
# quad1 = np.sum(_lanczos.quadrature(a, b, n, 0).prod(axis=1))
# quad2 = np.sum(_lanczos.quadrature(a, b, n, 1).prod(axis=1))
# assert np.isclose(quad1, quad2, atol=quad1*0.50)

def test_orthogonalization():
  from primate.diagonalize import lanczos
  np.random.seed(1234)
  n = 100
  A_sparse = csc_array(symmetric(n))
  v0 = np.random.uniform(size=n)
  from scipy.linalg import eigh_tridiagonal
  ew_true = np.linalg.eigh(A_sparse.todense())[0]

  rr_errors = {}
  for i in range(30):
    rr_errors[i] = []
    v0 = np.random.uniform(size=n)
    for j in range(0, n):
      (a,b) = lanczos(A_sparse, v0=v0, rtol=1e-8, orth=j, return_basis = False)
      ew_test = eigh_tridiagonal(a,b,eigvals_only=True)
      rr_errors[i].append(np.linalg.norm(np.sort(ew_test) - np.sort(ew_true)))

  ## Up to 5, results are identical, then through about 35 the error increases on average 
  ## only to really start descending around 50; past ~90, error is near machine precision
  from bokeh.plotting import figure, show 
  from bokeh.io import output_notebook
  output_notebook()
  p = figure(width=350, height=250)
  for i in range(30):
    p.line(np.arange(len(rr_errors[i])), rr_errors[i])
  show(p)