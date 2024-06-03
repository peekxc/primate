import numpy as np
import ssgetpy
from ssgetpy import search, fetch

from collections import namedtuple
MatrixRecord = namedtuple("MatrixRecord", ["matrix_norm", "spectral_gap", "rank", "nullity"])

def download_ssmc(matrix_list):
  assert isinstance(matrix_list, ssgetpy.matrix.MatrixList), "Invalid matrix list given"
  from scipy.io import loadmat
  import requests
  import pandas as pd
  SP = []
  for M in matrix_list:
    url = M.url().replace("MM/", "")[:-7]
    html = requests.get(url).content
    df_list = pd.read_html(html)
    di = df_list[2].to_dict()['SVD Statistics.1']
    sp_meta = MatrixRecord(float(di[0]), float(di[1]), int(di[3]), int(di[5]))
    file_lst = M.download(format="MAT")
    sp_mat = loadmat(file_lst[0])['Problem'][0][0][1]
    SP.append((sp_mat, sp_meta))
  return SP

mat_results = ssgetpy.search(isspd=False, dtype="real", rowbounds=(10, 1e5), limit=20)
sp_mats = download_ssmc(mat_results)

# eigvalsh_tridiagonal

# %% Parameter optimization 
import optuna


from primate.functional import numrank
from primate.trace import hutch

M, sp_info = sp_mats[3]
sw = np.linalg.svd(M.todense())[1]
max_b = min(sw[~np.isclose(sw, 0.0)])
min_a = min(sw)

from primate.functional import estimate_spectral_radius
estimate_spectral_radius(M, rtol = 0.0001, full_output=True)

def objective(trial):
  a = trial.suggest_float('a', 0.10 * min_a, 100 * min_a)
  b = trial.suggest_float('b', a, 10 * max_b)
  maxiter = trial.suggest_int('maxiter', 10, M.shape[0]*10)
  deg = trial.suggest_int('deg', 3, M.shape[0]-2)
  orth = trial.suggest_int('orth', 0, deg-2)
  nr, info = hutch(M, fun="smoothstep", a=a, b=b, deg=deg, maxiter=maxiter, orth=orth, ncv=orth+3, info=True)
  return (nr - sp_info.rank)**2 # + info['total_time_us']

study = optuna.create_study()
study.optimize(objective, n_trials=200)
study.trials
hutch(M, fun="smoothstep", **study.best_params)
# hutch(M, fun="smoothstep", deg=16, orth=15, ncv=16,seed=1234)


from optuna.visualization import plot_parallel_coordinate, plot_contour, plot_optimization_history, plot_intermediate_values
plot_parallel_coordinate(study)
plot_contour(study)
plot_optimization_history(study)
# plot_intermediate_values(study)


# hutch(M, fun="smoothstep", a=25449.6609699452, b=1e-9, maxiter=1200)

numrank(M)


np.flatnonzero(np.array([mr.nullity for M, mr in sp_mats]) > 0)

M = sp_mats[84][0]
gap = sp_mats[84][1].spectral_gap
# np.linalg.matrix_rank(sp_mats[84][0].todense()) # 2898 

from primate.random import symmetric
from primate.functional import estimate_spectral_gap
out = estimate_spectral_gap(M, rtol=0.0001, full_output=True)

# ew = np.random.uniform(size=100, low=0, high=130)
# ew[np.random.choice(np.arange(100), size=20)] = 0
# M = symmetric(100, pd=False, ew=ew)
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
k = 470
n = M.shape[1]
v0 = np.random.uniform(size=n)
v0 /= np.linalg.norm(v0)
(a,b) = lanczos(M, v0=v0, deg=40, orth=3, return_basis=False)
# rw, rv = eigh_tridiagonal(a,b,select='i', select_range=(0,0))
rw = eigvalsh_tridiagonal(a,b)
print(rw)


# M.data /= max(ew)
# np.sort(ew)
# (np.sort(ew)/max(ew))[19]

minimize(Rayleigh, jac=True, x0=np.ravel(rv), method="CG")
minimize(Rayleigh, jac=True, x0=np.ravel(rv), method="L-BFGS-B")
minimize(Rayleigh, jac=True, x0=np.ravel(rv), method="TNC")

## This works well iff M is positive definite
eigsh(M, k=1, sigma=0.0, which='LM', return_eigenvectors=False)

from scipy.optimize import minimize
def Rayleigh(u: np.ndarray):
  return (u[:,np.newaxis].T @ M @ u[:,np.newaxis]) / (np.linalg.norm(u)**2)

def Rayleigh_grad(u: np.ndarray):
  return np.ravel((2 / (np.linalg.norm(u)**2)) * (M @ u[:,np.newaxis] - Rayleigh(u) * u[:,np.newaxis]))

def Rayleigh(u: np.ndarray):
  u_dot = np.dot(u, u)
  uc = u[:,np.newaxis]
  y = M @ uc
  obj = (uc.T @ y) / u_dot
  gra = np.ravel((2 / u_dot) * (y - obj * uc))
  return obj, gra

u = np.random.uniform(size=M.shape[1])
minimize(Rayleigh, jac=True, x0=u, method="BFGS")
minimize(Rayleigh, jac=True, x0=u, method="Newton-CG")

## Best three: CG < L-BFGS-B < TNC
minimize(Rayleigh, jac=True, x0=u, method="CG") 
minimize(Rayleigh, jac=True, x0=u, method="L-BFGS-B") # L-BFGS-B is good 
minimize(Rayleigh, jac=True, x0=u, method="TNC") # TNC is better 

from primate.trace import hutch, hutchpp
from primate.functional import numrank
from primate.special import smoothstep

numrank(M, gap=1e-2, verbose=True, plot=False, deg=100)

hutch(M, fun=np.sign)

hutch(M, fun=smoothstep(a=1e-10,b=1e-6), deg=5, orth=3, ncv=5, maxiter=2000, plot=False, verbose=True)

hutchpp(M, fun=smoothstep(a=0.1, b=1.0), nb=M.shape[0], maxiter=200)
ew = np.linalg.eigvalsh(M.todense())
from scipy.sparse.csgraph import structural_rank
structural_rank(M)

# np.linalg.eigh(M.todense())[0]
ew = np.linalg.eigh(M.todense())[0]

from sksparse.cholmod import cholesky_AAt 
F = np.sort(cholesky_AAt(M, beta=0).D())
threshold = np.max(F) * max(M.shape) * np.finfo(np.float32).eps
true_threshold = max(ew) * max(M.shape) * np.finfo(M.dtype).eps


import bokeh 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
p = figure()
# p.scatter(np.arange(len(ew)), ew)
p.scatter(np.arange(len(F)), F)
show(p)

ew / max(ew)




