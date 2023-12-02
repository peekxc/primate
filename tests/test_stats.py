import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.sparse import csc_array, csr_array
from more_itertools import * 

## Add the test directory to the sys path 
import sys
import primate
ind = [i for i, c in enumerate(primate.__file__) if c == '/'][-3]
sys.path.insert(0, primate.__file__[:ind] + '/tests')

def test_incremental_est():
  np.random.seed(1234)
  x = np.random.uniform(size=10)
  mu, vr = np.zeros(len(x)), np.zeros(len(x))
  mu_prev, vr_prev = 0.0, 0.0
  for i, n in enumerate(range(1, len(x) + 1)):
    mu[i] = (x[i] + (n-1) * mu_prev) / n
    L = (n-2)/(n-1) if (n-2) > 0 else 0
    mu_prev = mu[i] if i == 0 else mu_prev # update before variance estimate
    vr[i] = L * vr_prev + (1.0 / n) * (x[i] - mu_prev)**2
    mu_prev = mu[i]
    vr_prev = vr[i]
  mu_true = np.cumsum(x) / np.arange(1, len(x)+1)
  vr_true = np.array([np.std(x[:i], ddof=1)**2 if i > 1 else 0.0 for i in range(1, len(x)+1)])
  assert np.allclose(mu_true, mu)
  assert np.allclose(vr_true, vr)

def test_CLT():
  from scipy.special import erfinv
  from scipy.stats import norm
  from primate.stats import sample_mean_cinterval
  np.random.seed(1234)
  x = np.random.normal(size=1500, loc=5.0, scale=3)
  sq_n, dof = np.sqrt(len(x)), len(x) - 1
  mu_est, sd_est = np.mean(x), np.std(x, ddof=1)
  
  z = np.sqrt(2) * erfinv(0.95)
  margin_of_error = z * sd_est / sq_n  
  ci_test = np.array([mu_est - margin_of_error, mu_est + margin_of_error])
  ci_true = norm.interval(0.95, loc = mu_est, scale= sd_est / sq_n)
  ci_lib = sample_mean_cinterval(x, conf=0.95, sdist='normal')
  assert np.allclose(ci_test, ci_true)
  assert np.allclose(ci_lib, ci_test)

  # atol = 0.50
  # for i in range(2, 50):
  #   mu = np.mean(x[:i])
  #   ci = sample_mean_cinterval(x[:i], conf=0.99, sdist='normal')
  #   print(f"{i}: est = {mu:.4}, |x - mu| = {(mu - 5.0):.4}, conf len: {np.take(np.diff(ci),0):.4}, CI says converged? { 0.50*np.diff(ci) <= atol }, actually converged? {(mu - 5.0) <= atol}")

def test_suggest():
  from primate.stats import suggest_nv_trace
  assert suggest_nv_trace(0.95, eps = 0.1) == 2213


