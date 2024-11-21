import numpy as np


# dtrtri
def update_trinv(B_inv: np.ndarray, b: np.ndarray):
	r"""Updates a square, upper-triangular inverse matrix with a new column.

	Given an inverse matrix $B^{-1}$ of some upper-triangular matrix $B$, this function computes the inverse:

	$$ B_\ast^{-1} = [B, b]^{-1}  $$

	where $[B, b]^{-1}$ denotes the inverse of matrix given by appending $B$ with the column vector $b$.

	"""
	n, m = B_inv.shape
	assert n == m and len(b) == (n + 1), "B must be n x n and `b` must have length `n + 1`"
	b = np.atleast_2d(b).reshape((n + 1, 1))
	assert B_inv.dtype == b.dtype, "dtypes of `B_inv` and `b` did not match."

	B_ast = np.zeros(shape=(n + 1, n + 1))
	B_ast[:n, :n] = B_inv
	B_ast[n, n] = np.reciprocal(b[-1]).item()
	B_ast[:n, [-1]] = B_ast[n, n] * ((-B_inv) @ b[:-1])
	return B_ast
