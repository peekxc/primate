import numpy as np
from scipy.sparse import load_npz
from primate.functional import spectral_density

def test_spectral_density():
  A = load_npz("/Users/mpiekenbrock/spirit/tests/UpLaplacian1_446_rank396.npz")
  from primate.functional import estimate_spectral_radius
  sr = estimate_spectral_radius(A)
  A.data = A.data / sr

  true_ew = np.linalg.eigh(A.todense())[0]
  tol = A.shape[0] * np.finfo(A.dtype).eps
  true_gap = np.min(true_ew[true_ew > tol])
  spectral_density(A, fun="smoothstep", a=1e-6, b=true_gap)
