import numpy as np 
# import cupy as cp
import numba as nb 
from numba import float32, float64, int32, int64, cuda

def poly(x: np.ndarray, mu_sqrt_rec: float64, a: np.ndarray, b: np.ndarray, z: np.ndarray, n: int64):
  z[0] = mu_sqrt_rec
  z[1] = (x - a[0]) * z[0] / b[1]
  for i in range(2, n):
    s = (x - a[i-1]) / b[i]
    t = -b[i-1] / b[i]
    z[i] = s * z[i-1] + t * z[i-2]
  
## Algorithm from: "Computing Gaussian quadrature rules with high relative accuracy." Numerical Algorithms 92.1 (2023): 767-793.
## By Laudadio, Teresa, Nicola Mastronardi, and Paul Van Dooren. 
def FTTR_weights(theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray, k: int64, weights: np.ndarray):
  n = len(alpha)
  mu_0 = np.sum(np.abs(theta[:k]))
  mu_sqrt_rec = 1.0 / np.sqrt(mu_0)
  p = np.zeros(n)
  for i in range(k):
    poly(theta[i], mu_sqrt_rec, alpha, beta, p, n)
    weight = 1.0 / np.sum(np.square(p))
    weights[i] = weight / mu_0
