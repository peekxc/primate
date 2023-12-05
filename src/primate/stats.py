import numpy as np
from typing import * 
from numbers import Number
import scipy.stats as st

## Based on Theorem 4.3 and Lemma 4.4 of Ubaru
def suggest_nv_trace(p: float, eps: float, f: str = "identity", dist: str = "rademacher") -> int:
  """Suggests a number of sample vectors to use to get an eps-accurate trace estimate with probability p."""
  assert p >= 0 and p < 1, "Probability of success 'p' must be  must be between [0, 1)"
  eta = 1.0 - p
  if f == "identity":
    return int(np.round((6 / eps ** 2) * np.log(2 / eta)))
  else: 
    raise NotImplementedError("TODO")

# def suggest_nv_mf(p: float, eps: float, max_f: float, k: int, e_min: float, e_max: float, n: int, dist: str = "rademacher"):
#   k = e_max / e_min 
#   rho = (np.sqrt(k) + 1)/(np.sqrt(k) - 1)
#   C = ((e_max - e_min) * (np.sqrt(k) + 1)**2 * max_f) / (2 * np.sqrt(k))  
#   mf_tr_ub = n * C / (rho ** 2*k) ## upper-bound on trace of matrix function


## See also: 
## https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
## https://cran.r-project.org/web/packages/distributions3/vignettes/one-sample-t-confidence-interval.html
def sample_mean_cinterval(a: np.ndarray, conf=0.95, sdist: str = ["t", "normal"]) -> tuple:
  """Confidence intervals for the sample mean of a set of measurements."""
  assert isinstance(conf, Number) and conf >= 0.0 and conf <= 1.0, "Invalid confidence measure"
  if sdist == ["t", "normal"] or sdist == "t":
    from scipy.stats import t
    mean, sem, m = np.mean(a), st.sem(a, ddof=1), t.ppf((1+conf)/2., len(a)-1)
    return mean - m*sem, mean + m*sem
  elif sdist == "normal": 
    from scipy.stats import norm
    sq_n = np.sqrt(len(a))
    mean, std = np.mean(a), np.std(a, ddof=1)
    return norm.interval(conf, loc=mean, scale=std / sq_n)
  else: 
    raise ValueError(f"Unknown sampling distribution '{sdist}'.")


## Manual approach
# sq_n, ssize = np.sqrt(len(a)), (len(a)-1)  
# s = np.std(a, ddof=1) # == (1.0 / np.sqrt(ssize)) * np.sum((a - mean)**2))
# rem = (1.0 - conf) / 2.0
# upper = st.t.ppf(1.0 - rem, ssize)
# lower = np.negative(upper)
# c_interval = mean + np.array([lower, upper]) * s / sq_n
# np.sqrt(2) * erfinv(2*0.025 - 1)
