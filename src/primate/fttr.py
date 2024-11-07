import numpy as np


# pythran export ortho_poly(float, float, float[:], float[:], float[:], int)
def ortho_poly(x: float, mu_sqrt_rec: float, a: np.ndarray, b: np.ndarray, z: np.ndarray, n: int):
	z[0] = mu_sqrt_rec
	z[1] = (x - a[0]) * z[0] / b[1]
	for i in range(2, n):
		s = (x - a[i - 1]) / b[i]
		t = -b[i - 1] / b[i]
		z[i] = s * z[i - 1] + t * z[i - 2]


## Algorithm from: "Computing Gaussian quadrature rules with high relative accuracy."
## Numerical Algorithms 92.1 (2023): 767-793, by Laudadio, Teresa, Nicola Mastronardi, and Paul Van Dooren.
# pythran export fttr(float[:], float[:], float[:], int, float[:])
def fttr(theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray, k: int, weights: np.ndarray):
	"""Forward three-term recurrence for computing Gaussian quadrature weights.

	This functions computes the weights of the Gaussian quadrature of nodes `theta` w.r.t a basis of orthogonal polynomials.
	"""
	n = len(alpha)
	mu_0 = np.sum(np.abs(theta[:k]))
	mu_sqrt_rec = 1.0 / np.sqrt(mu_0)
	p = np.zeros(n)
	for i in range(k):
		ortho_poly(theta[i], mu_sqrt_rec, alpha, beta, p, n)
		weight = 1.0 / np.sum(np.square(p))
		weights[i] = weight / mu_0
