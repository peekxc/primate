# %% Imports
import numpy as np
import imate
from pyimate.trace import trace_estimator

# %% Start with a toeplitz
A = imate.toeplitz(2, 1, size=500, gram=True) # gram = True for symmetric
tr_est, info = trace_estimator(A, info=True, num_threads=2)


# imate.schatten(A, method='slq')

## Smoothstep can indeed do well at detecting the rank!
tr_est = trace_estimator(A, matrix_function="smoothstep", a=1.5, b=1.51, orthogonalize=0, plot=False, info=False)

np.sum(np.linalg.eigvalsh(A.todense()) >= 1.5)
np.linalg.matrix_rank(A.todense())

# %% benchmarks
import timeit
from scipy.sparse import random
A = random(2500, 2500, density=0.05**2, format='csr')
B = A.T @ A
print(f"nnz %: {B.nnz/(B.shape[0]**2)}")
B_dense = B.todense()

# timeit.timeit(lambda: B @ np.random.uniform(size=B.shape[0]), number=1000)

timeit.timeit(lambda: np.sum(np.linalg.eigvalsh(B_dense)), number=5)
timeit.timeit(lambda: trace_estimator(B, orthogonalize=0, min_num_samples=20, max_num_samples=30), number=5)

# %% Investigate slowness
import timeit
A = imate.toeplitz(2, 1, size=500, gram=True) # gram = True for symmetric
tr_est, info = trace_estimator(A, info=True)

## For 30 samples, orthogonalize == 1: 
## (1) timeit estimates 500x500 toeplitz Gram matrix averages 0.3940974158666677 seconds to compute
## (2) Mean reported alg. wall time for 15 iterations is about 0.408502197265625 seconds, so allocation probably not an issue
## (3) Optimizing matvec w/ Map ==> 0.00402 seconds for (1) and 0.00367 for (2)
## (4) Optimizing eigenvector allocation ==> 0.00399 for (1) and 0.00322 for (2)
timeit.timeit(lambda: trace_estimator(A, orthogonalize=1, min_num_samples=30, max_num_samples=30), number=15)/15
np.mean([trace_estimator(A, orthogonalize=1, min_num_samples=30, max_num_samples=30, info=True)[1]['time']['alg_wall_time'] for _ in range(15)])




# %% More testing
A = imate.toeplitz(np.random.uniform(20), np.random.uniform(19))
#  orthogonalize=20, confidence_level=0.99, error_atol=0.0, error_rtol=1e-6, min_num_samples=150, max_num_samples=200
tr_est, info = trace_estimator(A, p=1.0, orthogonalize=3, lanczos_degree=20, confidence_level=0.90, error_rtol=1e-2, min_num_samples=50, max_num_samples=200, num_threads=2, verbose=False)
tr_est_, info_ = imate.trace(A, gram=False, method="exact", return_info=True)
tr_true = np.sum(A.diagonal())
print(f"True: {tr_true:.8f}, Est: {tr_est[0]:.8f}, Exact imate: {tr_est_:.8f}")

## Adjusted to remove outliers
s_samples = np.sort(np.ravel(info['convergence']['samples']))
s_samples = s_samples[~np.isnan(s_samples)]
n_outliers = int(info['convergence']['num_outliers'])
nonoutliers = np.argsort(np.argsort(np.abs(s_samples - np.mean(s_samples))))[n_outliers:]
print(f"outlier adjusted: {np.mean(s_samples[nonoutliers]):.8f}")


# %% 
import bokeh 
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.models import Span
output_notebook()

# %% Plot the actual samples 
samples = np.ravel(info['convergence']['samples'])
p = figure(width=200, height=200, title="Trace sample estimates")
p.scatter(np.arange(len(samples)), samples, size=2.5)
p.add_layout(Span(location=np.sum(A.diagonal()), dimension='width', line_color='red', line_width=1.5))
p.add_layout(Span(location=tr_est[0], dimension='width', line_color='blue', line_width=1.5))
show(p)





# %% Debug: Where is the bias in the trace estimate coming from
# from scipy.linalg import eigh_tridiagonal
# from scipy.sparse.linalg import aslinearoperator
# from pyimate import _diagonalize
# n = A.shape[0]
# v0 = np.random.uniform(size=A.shape[1])
# alpha, beta = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

# A_lo = aslinearoperator(A)
# _diagonalize.lanczos_tridiagonalize(A_lo, v0, 1e-8, n-1, alpha, beta)
# ew = np.sort(eigh_tridiagonal(alpha, beta[:-1], eigvals_only=True))

# np.sum(ew)

# tol[i] = np.mean(np.abs(ew - np.sort(eigsh(A, k=n, return_eigenvectors=False))))

# info['convergence']['samples'].mean(axis=1)
# n_samples = int(info['convergence']['num_samples_used'])
# np.cumsum(np.ravel(info['convergence']['samples']))/np.arange(1, n_samples+1)
# %% Diagnostics
from imate._trace_estimator import trace_estimator_plot_utilities as te_plot
from imate._trace_estimator import trace_estimator_utilities as te_util 

te_util.print_summary(info)
te_plot.plot_convergence(info)




# %% Try to match imate
A = imate.toeplitz(2, 1, size=1000000, gram=True)
trace_estimator(A, 
  min_num_samples=20, max_num_samples=80, error_rtol=2e-4, confidence_level=0.95, outlier_significance_level=0.001, 
  num_threads = 1, 
  plot=True
)
# %% 
from imate import toeplitz, trace, schatten
A = toeplitz(2, 1, size=100, gram=True)
trace(A, gram=False, p=2.5, method='slq')
schatten(A, method="slq")
