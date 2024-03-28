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

mat_results = ssgetpy.search(isspd=True, dtype="real", rowbounds=(10, 1e5), limit=10)
sp_mats = download_ssmc(mat_results)


M = sp_mats[0][0]
gap = sp_mats[0][1].spectral_gap
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




