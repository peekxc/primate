import numpy as np

%load_ext pythran.magic

%%pythran 
#pythran export lanczos(float32[:, :], float32 [:], int, float)
def lanczos(A: np.ndarray, v0: np.ndarray, k: int, tol: float) -> int:
	"""Base lanczos algorithm, for establishing a baseline"""
	n = A.shape[0]
	v0 = np.random.uniform(size=A.shape[1], low=-1.0, high=+1.0) if v0 is None else np.array(v0)
	assert k <= n, "Can perform at most k = n iterations"
	assert len(v0) == A.shape[1], "Invalid starting vector; must match the number of columns of A."
	alpha = np.zeros(n + 1, dtype=np.float32)
	beta = np.zeros(n + 1, dtype=np.float32)
	qp = np.zeros(n, dtype=np.float32)
	qc = v0.copy()
	qc /= np.linalg.norm(v0)
	for i in range(k):
		qn = A @ qc - beta[i] * qp
		alpha[i] = np.dot(qn, qc)
