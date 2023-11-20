# %%
import numpy as np
from scipy.io import loadmat

np.random.seed(1234)
ew = np.random.uniform(size=150, low=0, high=2)
Q, R = np.linalg.qr(np.random.uniform(size=(len(ew), len(ew))))
A = Q @ np.diag(ew) @ Q.T
A = (A + A.T) / 2
assert np.all(A == A.T)

# X = loadmat("/Users/mpiekenbrock/XTrace/X.mat")["X"]
# Om = loadmat("/Users/mpiekenbrock/XTrace/Om.mat")["Om"]
# print(X.trace())
# xtrace(X)

# Y = X @ Om
# Q, R = np.linalg.qr(Y, "reduced")
# Z = A @ Q

# sample_isotropic(8, 1, method="improved")


# %% 
def sample_isotropic(n: int, m: int, method: str = "rademacher"):
	from primate.random import rademacher
	if method == "rademacher":
		return rademacher(size=(n, m))
	elif method == "improved":
		W = np.random.normal(size=(n, m))
		W = np.sqrt(n) * (W / np.linalg.norm(W, axis=0))  ## confirmed
		return W
	elif method == "sphere":
		W = np.random.normal(size=(n, m))
		W = np.sqrt(n) * (W @ np.diag(1.0 / np.linalg.norm(W, axis=0)))
		return W
	else:
		raise ValueError("invalid method supplied")

def __xtrace(W: np.ndarray, Z: np.ndarray, Q: np.ndarray, R: np.ndarray, method: str):
	"""Helper for xtrace function"""
	from scipy.linalg import solve_triangular
	diag_prod = lambda A, B: np.diag(A.T @ B)[:, np.newaxis]

	## Invert R
	n, m = W.shape
	W_proj = Q.T @ W
	R_inv = solve_triangular(R, np.identity(m)).T # to replace with dtrtri
	S = (R_inv / np.linalg.norm(R_inv, axis=0)) # this is cnormc

	## Handle the scale
	if not method == 'improved':
	  scale = np.ones(m)[:,np.newaxis] # this is a column vector
	else:
	  col_norm = lambda X: np.linalg.norm(X, axis=0)
	  c = n - m + 1
	  scale = c / (n - (col_norm(W_proj)[:,np.newaxis])**2 + (diag_prod(S,W_proj) * col_norm(S)[:,np.newaxis])**2)

	## Intermediate quantities
	H = Q.T @ Z
	HW = H @ W_proj
	T = Z.T @ W
	dSW, dSHS = diag_prod(S, W_proj), diag_prod(S, H @ S)
	dTW, dWHW = diag_prod(T, W_proj), diag_prod(W_proj, HW)
	dSRmHW, dTmHRS = diag_prod(S, R - HW), diag_prod(T - H.T @ W_proj, S)

	## Trace estimate
	tr_ests = np.trace(H) * np.ones(shape=(m, 1)) - dSHS
	tr_ests += (-dTW + dWHW + dSW * dSRmHW + abs(dSW) ** 2 * dSHS + dTmHRS * dSW) * scale
	t = tr_ests.mean()
	err = np.std(tr_ests) / np.sqrt(m)
	return t, tr_ests, err


def xtrace(A, error_atol: float = 0.1, error_rtol: float = 1e-6, nv: int = 1, cond_tol: float = 1e8, method: str = "improved"):
	assert error_atol >= 0.0 and error_rtol >= 0.0, "Error tolerances must be positive"
	assert cond_tol >= 0.0, "Condition number must be non-negative"
	n = A.shape[0]
	Y, Om, Z = np.zeros(shape=(n, 0)), np.zeros(shape=(n, 0)), np.zeros(shape=(n, 0))
	t, err = np.inf if (error_rtol != 0) else 0, np.inf
	it = 0
	while Y.shape[1] < A.shape[1]:  # err >= (error_atol + error_rtol * abs(t)):
		ns = max(nv,2) if it == 0 else nv 
		ns = min(A.shape[1] - Y.shape[1], ns)
		NewOm = sample_isotropic(n, ns, method=method)
		
		## If the sampled vectors have a lot of linear dependence  won't be (numerically) span a large enough subspace
		## to permit sufficient exploration of the eigen-space, so we optionally re-sample based on a loose upper bound
		## Based on: https://math.stackexchange.com/questions/1191503/conditioning-of-triangular-matrices
		cond_numb_bound = np.inf
		while cond_numb_bound > cond_tol:
			tmp_Y, tmp_Om = np.c_[Y, A @ NewOm], np.c_[Om, NewOm]
			Q, R = np.linalg.qr(tmp_Y, "reduced") 
			R_mass = np.abs(np.diag(R))
			_cn = 3 * np.max(R_mass) / np.min(R_mass)
			cond_numb_bound = 0.0 if np.isclose(_cn, cond_numb_bound) else _cn
			# print(cond_numb_bound)
		Y, Om = tmp_Y, tmp_Om 
		Z = np.c_[Z, A @ Q[:, -ns:]]
		t, t_samples, err = __xtrace(Om, Z, Q, R, method)
		it += 1
		print(t)
	return t, t_samples, err

# savemat("/Users/mpiekenbrock/XTrace/bad_Om.mat", { 'Om' : NewOm })
# %%
