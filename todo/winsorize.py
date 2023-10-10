from scipy.linalg import toeplitz
from primate.trace import slq
import numpy as np

## Attempt at improving the robustness of the trace estimator 
## tried winsorization using a weibull fit but it seemed unstable, expensive, and not very powerful in terms of improvement
def weibull_mean_winsorize(samples: np.ndarray, r: int = 1):
  from scipy.stats import weibull_min
  assert r >= 1, "Must have at least r+1 element to winsorize"
  k = r+1
  samples_part = np.partition(samples, kth=[k,-k])
  low, hgh = np.sort(samples_part[:k]), np.sort(samples_part[-k:])
  samples_winsor = samples.copy()
  samples_winsor[samples_winsor <= low[-1]] = low[-1]
  samples_winsor[samples_winsor >= hgh[0]] = hgh[0]
  weibull_params = weibull_min.fit(samples_winsor)
  return weibull_min.mean(*weibull_params)


slq_params = dict(gram=True, orthogonalize=10, confidence_level=0.95, error_atol=1e-6, min_num_samples=150, max_num_samples=200, num_threads=1)
np.random.seed(1234)
X = toeplitz(np.random.uniform(size=50))
A = X.T @ X
tr_true = np.sum(A.diagonal())
slq_params["max_num_samples"] = 50000
tr_est_1 = slq(A, distribution="normal", rng_engine="pcg", plot = True, **slq_params)

tr_samples = np.squeeze(tr_est_1[1]['convergence']['samples'])
tr_sample_avgs = np.cumsum(tr_samples)/np.arange(1, 50001)

print(f"Trace true: {tr_true}")
low_outliers = tr_samples <= np.quantile(tr_samples, 0.07)
hgh_outliers = tr_samples >= np.quantile(tr_samples, 0.99)
np.mean(tr_samples[~np.logical_or(low_outliers, hgh_outliers)])

from scipy.stats.mstats import winsorize
np.mean(winsorize(tr_samples, limits=(0.0, 1.0)))

import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import Span
output_notebook()

ind = np.arange(1, len(tr_samples)+1) 
p = figure(width=400, height=250)
p.scatter(ind[:500], tr_samples[:500], size=1.5)
p.line(ind[:500], tr_sample_avgs[:500], color='red', line_width=1.50)
tr_span = Span(location=tr_true, dimension='width', line_color='black', line_width=1.5)
p.add_layout(tr_span)
show(p)
