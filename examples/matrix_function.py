# %% 
import numpy as np 
from primate.diagonalize import lanczos
from scipy.linalg import eigh_tridiagonal
from primate.random import rademacher


# %% Start with a matrix with well-separated eigenvalues
np.random.seed(1234)
ew = 0.2 + 1.5*np.linspace(0, 5, 15)
Q,R = np.linalg.qr(np.random.uniform(size=(15,15)))
A = Q @ np.diag(ew) @ Q.T
A = (A + A.T) / 2
assert np.allclose(np.linalg.eigvalsh(A) - ew, 0)


# %% Trace estimation by approximating the matrix function
v0 = rademacher(A.shape[1])
(a,b), Q = lanczos(A, v0, max_steps=A.shape[1], orth=0, return_basis=True)
rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
rw = rw / (rw + 1e-4)
y_test = np.linalg.norm(v0) * (Q @ V @ (V[0,:] * rw))


ew, ev = np.linalg.eig(A)
y_true = (ev @ np.diag((ew / (ew + 1e-4))) @ ev.T) @ v0
assert np.linalg.norm(y_test - y_true) < 1e-3
assert np.isclose(np.sum((ew / (ew + 1e-4))), np.dot(v0, y_true), atol=1e-4)
