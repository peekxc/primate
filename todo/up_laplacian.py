import numpy as np
from itertools import combinations
def up_laplacian_matvec(x, v, degree, simplices, I, J):
	d = simplices.shape[1]
	sgn_pattern = ([-1,1]*d)[:d]
	x = x.reshape(-1)
	v.fill(0)
	v += degree * x

	# for s_ind, s in enumerate(simplices):
	# 	for (f1, f2), sgn_ij in zip(combinations(s, 2), sgn_pattern):
	# # 		ii, jj = self.index(f1), self.index(f2)
	# 		v[ii] += sgn_ij * x[jj] * wfl[ii] * ws[s_ind] * wfr[jj]
	# 		v[jj] += sgn_ij * x[ii] * wfl[jj] * ws[s_ind] * wfr[ii]
	# for s_ind, s in enumerate(self.simplices):
	# 	for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
	# 		ii, jj = self.index(f1), self.index(f2)
	# 		self._v[ii] += sgn_ij * x[jj] * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
	# 		self._v[jj] += sgn_ij * x[ii] * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
	return v

#pythran export up_laplacian_matvec(float[], float[], float[], int[:,:])

