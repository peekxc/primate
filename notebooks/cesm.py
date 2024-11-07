# %% Imports
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from primate.quadrature import lanczos_quadrature, spectral_density
from primate.stochastic import symmetric
from primate.lanczos import lanczos, OrthogonalPolynomialBasis
from primate.stochastic import symmetric
from primate.quadrature import spectral_density
from landmark import landmarks

output_notebook()

# %%
rng = np.random.default_rng(1234)
xx = rng.uniform(size=35, low=0, high=1)
ew = np.sort(xx[landmarks(xx[:, np.newaxis], 25)])


# %%
