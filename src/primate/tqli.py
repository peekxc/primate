import numpy as np


# pythran export sign(float, float)
def sign(a: float, b: float):
	# return 1 if b > 0 else (-1 if a < 0 else 1)
	return int(b > 1) - int(a < 0) + 1


## Based on: https://github.com/rappoccio/PHY410/blob/f6a183eb48807841f6d35be45b4aa845d905a04c/cpt_python/cpt.py#L438
## This uses Givens rotations instead of divide-and-conquer to find the eigenvalues of a tridiagonal matrix
## Uses O(1) space, thus is preferred over Golub-Welsch if only ritz values are needed and space is an issue.
## However this is much slower and less stable than divide-and-conquer, if can fit in memory.
# pythran export tqli(float[:], float[:], float[:][:], int)
def tqli(d: np.ndarray, e: np.ndarray, Z: np.ndarray, max_iter: int = 30):
	"""Tridiagonal QL Implicit algorithm w/ shifts to determine the eigen-pairs of a real symmetric tridiagonal matrix.

	Note that with shifting, the eigenvalues no longer necessarily appear on the diagonal in order of increasing absolute magnitude.

	Based on pseudocode from the Chapter on "Eigenvalues and Eigenvectors of a Tridiagonal Matrix" in NUMERICAL RECIPES IN FORTRAN 77:
	THE ART OF SCIENTIFIC COMPUTING.

	Parameters:
	  d = Diagonal elements of the tridiagonal matrix.
	  e = Subdiagonal elements of the tridiagonal matrix.
	  max_iter = Number of iterations to align the eigenspace(s).
		Z:  Matrix to store eigenvectors. Can be empty array to avoid compute eigenvectors entirely.

	Returns:
	  w = The eigenvalues of the tridiagonal matrix T(d,e).
	"""
	assert len(d) == len(e), "Diagonal and subdiagonal should have same length (subdiagonal should be prefixed with 0)"
	assert np.isclose(e[0], 0.0), "Subdiagonal first element should be zero"
	n = len(d)
	e[:-1] = e[1:]
	e[n - 1] = 0.0
	for it in range(n):
		ii = 0
		m = it
		while True:
			## Look for a single small subdiagonal element to split the matrix
			for ml in range(it, n - 1):
				m = ml
				dd = abs(d[m]) + abs(d[m + 1])
				if (abs(e[m]) + dd) == dd:
					break
				else:
					m += 1
			# print(f"{it}: {m} vs. {m if it >= (n - 1) else it + np.argmax(np.isclose(np.abs(e[it:n]), 0.0, atol=1e-15))}")
			## Though the above is equivalent to the below, the fortran was just optimal in terms of flop count
			# m = it + np.argmax(np.isclose(np.abs(e[it:n]), 0.0, atol=1e-15))

			if m != it:
				if ii > max_iter or e[it] == 0.0:
					## Could throw here, but since this is used downstream in randomized algorithm we take as-is
					break
				ii += 1
				g = (d[it + 1] - d[it]) / (2.0 * e[it])  # shift
				r = np.hypot(g, 1.0)  # pythag
				g = d[m] - d[it]
				g += e[it] / (g + sign(r, g))  # dm - ks
				# g = d[m] - d[l] + e[l] if np.isclose(g + sign(r, g), 0.0) else (d[m] - d[l] + e[l] / (g + sign(r, g)))
				s, c, p = 1.0, 1.0, 0.0
				for i in range(m - 1, it - 1, -1):
					## Plane rotation followed by Givens to restore tridiagonal
					f, b = s * e[i], c * e[i]
					e[i + 1] = r = np.hypot(f, g)
					if r == 0.0:  ## Recover from underflow
						# if abs(r) <= 2.2250738585072014e-128:
						d[i + 1] -= p
						e[m] = 0.0
						break
					s, c = (f / r, g / r)
					g = d[i + 1] - p
					r = (d[i] - g) * s + 2.0 * c * b
					p = s * r
					d[i + 1] = g + p
					g = c * r - b
					if np.prod(Z.shape) > 0:
						for k in range(n):
							f = Z[k][i + 1]
							Z[k][i + 1] = s * Z[k][i] + c * f
							Z[k][i] = c * Z[k][i] - s * f
				if r == 0.0 and i >= it:
					continue
				d[it] -= p
				e[it] = g
				e[m] = 0.0
			if m == it:
				break
