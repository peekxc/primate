from geomcover.io import coo_array
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv, eigsh


def laplacian_matrix(A):
	degree = np.sum(A, axis=1)
	return np.diag(degree) - A


def effective_resistance(L_pinv, u, v):
	return L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]


def spectral_sparsifier(A, weights=None, epsilon: float = 0.1):
	n = A.shape[0]
	q = int(8 * n * np.log(n) / epsilon**2)
	L = laplacian_matrix(A)

	## Moore-Penrose pseudoinverse of L
	L_pinv = np.linalg.pinv(L)

	## Effective resistance for all edges
	U, V = np.array(np.nonzero(np.triu(A)))  # Get upper triangular edges
	resistances = L_pinv[U, U] + L_pinv[V, V] - 2 * L_pinv[U, V]

	## Probabilities proportional to w_e * R_e
	m = len(resistances)
	weights = np.ones(m) if weights is None else np.array(weights)
	probs = weights * resistances
	probs /= np.sum(probs)

	## Sample q edges with replacement
	rng = np.random.default_rng()
	EDGE_IND = np.ravel_multi_index((U, V), dims=A.shape)
	new_edges = np.zeros(m)
	batch_size = int(np.sqrt(q)) // 1
	for _ in range(q // batch_size):
		idx = rng.choice(m, size=batch_size, p=probs)
		uv_ind = np.ravel_multi_index((U[idx], V[idx]), dims=A.shape)
		new_edges[np.searchsorted(EDGE_IND, uv_ind)] += weights[idx] / (q * probs[idx])

	## Build the sampled adjacency matrix
	AS = np.zeros_like(A)
	AS.ravel()[EDGE_IND] = new_edges
	AS += AS.T
	return AS


from scipy.sparse import random_array, coo_array, csc_array

A = random_array((100, 100), density=0.025)
A = A @ A.T
(A.nnz / np.prod(A.shape)) * 100


G_sampled = spectral_sparsifier(A.todense(), epsilon=0.1)
G_sparse = coo_array(G_sampled)

from scipy.optimize import minimize

x = np.random.uniform(size=G_sparse.shape[0])
x @ A @ x
x @ G_sparse @ x

minimize()
