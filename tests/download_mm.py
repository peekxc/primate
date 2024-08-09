import numpy as np
import ssgetpy
from ssgetpy import search, fetch
import pathlib

from primate.operator import ShiftedOp

def infer_literal(s):
  """Attempts to convert the type of 's' into a natural base type, e.g. int, float, or str"""
  from ast import literal_eval
  if isinstance(s, str):
    try:
      val = literal_eval(s)
    except:
      val = s
  else:
    val = s
  val = int(val) if isinstance(val, float) and val.is_integer() else val
  return val
  
def download_ssmc(matrix_list, output_dir: str):
  assert isinstance(matrix_list, ssgetpy.matrix.MatrixList), "Invalid matrix list given"
  from scipy.io import loadmat, savemat
  from scipy.sparse import save_npz
  import pickle
  import requests
  import pandas as pd
  import os.path
  from collections import namedtuple
  MatrixRecord = namedtuple("MatrixRecord", ["matrix_norm", "spectral_gap", "rank", "nullity"])

  for M in matrix_list:
    url = M.url().replace("MM/", "")[:-7]
    html = requests.get(url).content
    df_list = pd.read_html(html)
    sp_name = df_list[0].to_dict()[1][0]
    out_file = output_dir + "/" + sp_name + ".npz"
    di = df_list[2].to_dict()
    if 'SVD Statistics.1' in di and not os.path.isfile(out_file):
      di = di['SVD Statistics.1']
      sp_meta = MatrixRecord(float(di[0]), float(di[1]), int(di[3]), int(di[5]))
      file_lst = M.download(format="MAT")
      sp_mat = loadmat(file_lst[0])['Problem'][0][0][1]
      sp_meta1 = dict(zip(df_list[0].to_dict()[0].values(), df_list[0].to_dict()[1].values()))
      sp_meta2 = dict(zip(df_list[1].to_dict()[0].values(), df_list[1].to_dict()[1].values()))
      sp_meta3 = {
        "matrix_norm" : float(di[0]), 
        "spectral_gap": float(di[1]), 
        "rank" : int(di[3]),
        "nullity": int(di[5])
      }
      sp_meta = sp_meta1 | sp_meta2 | sp_meta3
      sp_meta = { k : infer_literal(v) for k,v in sp_meta.items() }
      # savemat(out_file, {'matrix' : sp_mat, 'meta': sp_meta},  do_compression=True)
      np.savez_compressed(out_file, matrix=sp_mat, meta=sp_meta)


mat_results = ssgetpy.search(isspd=True, dtype="real", rowbounds=(10, 1e5), limit=250)
download_ssmc(mat_results, '/Users/mpiekenbrock/primate/data')


w = np.load("/Users/mpiekenbrock/primate/data/testing.npz", allow_pickle=True)


# %% 
npz_file = np.load("/Users/mpiekenbrock/primate/data/bcsstk05.npz", allow_pickle=True)
A, meta = npz_file['matrix'], npz_file['meta']
['matrix']


# %% Parameter optimization 
import optuna
import os
from primate.functional import numrank
from primate.trace import hutch
from primate.functional import spectral_density, spectral_gap
from primate.quadrature import sl_gauss

pd = []
npz_files = [fn for fn in os.listdir("/Users/mpiekenbrock/primate/data") if fn[-3:] == 'npz']
for npz in npz_files:
  npz_file = np.load("/Users/mpiekenbrock/primate/data/" + npz, allow_pickle=True)
  A, meta = npz_file['matrix'].item(), npz_file['meta'].item()
  if meta['Symmetric'].lower() == 'yes': # and meta['Positive Definite'].lower() == 'yes':
    pd.append((A, meta))

full_rank = np.array([meta['Num Rows'] == meta['rank'] for (A, meta) in pd])
print(f"Full rank: {np.sum(full_rank)}/{len(pd)}")
rank_deficient = [(A, meta) for (A,meta), is_fr in zip(pd, full_rank) if not is_fr]

## Load a rank deficient matrix 
from primate.functional import normalize_spectrum
A, meta = rank_deficient[2]
# A.data = np.sign(A.data)
A, sr = normalize_spectrum(A)
ew = np.linalg.eigh(A.todense())[0]
pos_ew = np.sort(ew)[-meta['rank']:]
gap = np.min(pos_ew)
print(f"True gap: {gap}, True rank: {meta['rank']}, shape: {A.shape}, interval: [{np.min(ew)}, {np.max(ew)}]")
from scipy.sparse import csr_array
# gap_est = spectral_gap(A, shortcut=True)  
# print(np.isclose(gap_est, gap))

## Maybe just try 


from scipy.sparse.linalg import eigsh
B = ShiftedOp(A)
# eigsh(B, k=1, which='LM', sigma=0.0)
# B.num_matvecs
# Stopped at 340k!

## lsmr is indeed better
from scipy.sparse.linalg import cg, cgs, lgmres, lsqr, lsmr
from primate.operator import ShiftedInvOp
x = np.random.uniform(size=B.shape[0])
# b = B @ x
b = np.ones(B.shape[0])
lsqr(B, b)
lsmr(B, b)
B.num_matvecs = 0
cgs(B, b)
cg(B, b)

op = ShiftedInvOp(A, sigma=1e-9, solver="lsmr")
eigsh(op, which='LM', k=1, sigma=op.sigma, OPinv=op, return_eigenvectors=False, tol=0).take(0)
op.num_matvecs
op.A_shift.num_matvecs


from primate.diagonalize import lanczos, rayleigh_ritz, eigh_tridiagonal
from scipy.sparse.linalg import aslinearoperator
# (a,b), Q = lanczos(B, deg=B.shape[0], return_basis = True)
# rw, Y = eigh_tridiagonal(a, b)
rw, Y, Q = rayleigh_ritz(B, deg=20, return_eigenvectors=True, return_basis=True)

Minv = Q @ Y @ np.diag(1/rw) @ Y.T @ Q.T
cgs(B, b, M=Minv)
lsmr(B, x)

# rw, Y = rayleigh_ritz(B, deg = 20, return_eigenvectors=True)

Minv = np.diag(np.reciprocal(rw))
def preconditioner(x):
  return Y @ Minv @ Y.T @ x

aslinearoperator(preconditioner)
cg(B, b, M=preconditioner)

## All the non-least-squares solvers returns junk (every element is +/- Ce+17) after 48750 (10*n) iterations
## unless b is actually in the image of A

from primate.diagonalize import rayleigh_ritz
rayleigh_ritz(A, deg=500)

# %% Try shift-invert tridiagonal 
from primate.random import symmetric
from primate.diagonalize import lanczos 
np.random.seed(1234)
ew = np.random.uniform(size=100, low=0, high=1.0)
ew[ew <= 0.10] = 0.0
# ew = np.array([0, 0, 0.001, 0.1, 0.5, 0.75, 0.75, 0.80, 0.90, 1.0])
A = symmetric(len(ew), pd=False, ew=ew) 

B = lanczos(A, deg=A.shape[0], sparse_mat=True)
from scipy.linalg import solve_triangular, eigvalsh_tridiagonal
from scipy.sparse.linalg import aslinearoperator, spsolve_triangular

from line_profiler import LineProfiler
op = ShiftedInvOp(B, sigma=1e-5, solver="lgmres")

profile = LineProfiler()
# profile.add_function(eigsh)
profile.add_function(hutch)
profile.add_function(op._matvec)
profile.enable_by_count()

profile.print_stats()

eigsh(op, which='LM', k=1, sigma=op.sigma, OPinv=op, return_eigenvectors=False, tol=0).take(0)

from primate.trace import hutch
hutch(A, fun="smoothstep", a=1e-9, b=9.78158977673704e-06, verbose=True, deg=1000, maxiter=10)

from scipy.linalg import eigh_tridiagonal

a,b = lanczos(A, deg=A.shape[0])
rw = rayleigh_ritz(A, deg=A.shape[0])
rw_min, rv_min = eigh_tridiagonal(a, b, eigvals_only=False, select='i', select_range=(2,2))

op = ShiftedInvOp(A, sigma=rw_min, solver="lgmres", maxiter=10)
eigsh(op, which='LM', k=1, sigma=op.sigma, OPinv=op, return_eigenvectors=False, tol=0.4).take(0)

op.num_matvecs

H = (A - rw_min*np.eye(A.shape[0])) @ (A - rw_min*np.eye(A.shape[0]))
from scipy.sparse.linalg import eigsh

from scipy.sparse.linalg._interface import _PowerLinearOperator
from primate.operator import ShiftedOp
from scipy.optimize import minimize_scalar
HS = ShiftedOp(A, sigma=rw_min.item())
HP = _PowerLinearOperator(HS, 2)
# eigsh(A, k=1, which='LM', return_eigenvectors=False)
omega = spectral_radius(HP)
B = ShiftedOp(HP, sigma=omega)
eigsh(B, k=1, which='LM', return_eigenvectors=False)

xp = spectral_radius(B, rtol=0.00001)
obj = lambda x: (xp - ((x - rw_min)**2 + omega))**2 
res = minimize_scalar(obj)
res.x

B, sr = normalize_spectrum(A)
hutch(B, fun="smoothstep", a=1e-9, b=6.79706e-07, deg=B.shape[0])


from scipy.sparse.linalg import lgmres, cg, cgs
_, V = eigh_tridiagonal(a, b, eigvals_only=False, select='i', select_range=(1,3))

cgs(A, b=rv_min)
cg(A, b=rv_min, rtol=0.01, atol=1e-5)

(rv_test, _) = lgmres(A, b=rv_min, x0=rv_min, rtol=0.01, atol=1e-5, outer_k=3, outer_v=[(v, None) for v in V.T])
rv_test /= np.linalg.norm(rv_test)
rw_test = np.linalg.norm(A @ rv_test)

diffs = A.dot( rv_test ) - rv_test * rw_test
maxdiffs = np.linalg.norm( diffs, axis=0, ord=np.inf )
print("|Av - λv|_max:", maxdiffs)


cg(A, )

import timeit
timeit.timeit(lambda: op @ np.random.uniform(size=op.shape[1]), number=10)
timeit.timeit(lambda: A @ np.random.uniform(size=op.shape[1]), number=10)


np.linalg.norm((A @ rv_min) - rw_min * rv_min)
# eigsh(aslinearoperator(A), which='LM', k=1, sigma=rw_min.item(), return_eigenvectors=False, tol=0.1, v0=rv_min.flatten())
# eigsh(aslinearoperator(A), which='LM', k=1, sigma=9.76243e-06, return_eigenvectors=False, tol=0.1)


# from primate.operator import ShiftedInvOp
# from scipy.sparse.linalg import eigsh, aslinearoperator
# # "bicg", "bicgstab", "cg", "cgs", "gmres", "lgmres", "minres", "qmr", "gcrotmk", "tfqmr"
# op = ShiftedInvOp(A, sigma=1e-6, solver="bicgstab", maxiter=20)
# eigsh(op, which='LM', k=1, sigma=op.sigma, OPinv=op, return_eigenvectors=False, tol=0.4).take(0)
    
# op.num_matvecs
# op.num_adjoint
# eigsh(aslinearoperator(A), k=1, which='LM', sigma=1e-2, return_eigenvectors=False, tol=0.4)
# nodes, weights = sl_gauss(A, n=50, deg=50).T

x = np.random.uniform(size=A.shape[0])
# x = np.random.choice([-1.0,1.0], size=A.shape[0])
x /= np.linalg.norm(x)
x @ A @ x



spectral_density(A, plot=False)

nodes, weights = sl_gauss(A, n=150, deg=20).T
np.min(nodes)

gap


p = figure()
p.scatter(np.arange(len(rdiffs)), rdiffs)
show(p)

# sw = np.linalg.svd(A.todense())[1]
max_b = min(sw[~np.isclose(sw, 0.0)])
min_a = min(sw)



# %% 
from primate.functional import estimate_spectral_radius
estimate_spectral_radius(A, rtol = 0.0001, full_output=True)

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




