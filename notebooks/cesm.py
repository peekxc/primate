# %% Imports
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from landmark import landmarks
from primate.lanczos import OrthogonalPolynomialBasis, lanczos
from primate.quadrature import lanczos_quadrature, spectral_density
from primate.random import symmetric

output_notebook()

# %%
rng = np.random.default_rng(1234)
xx = rng.uniform(size=35, low=0, high=1)
ew = np.sort(xx[landmarks(xx[:, np.newaxis], 25)])


# %%
from primate.plotting import figure_csm

show(figure_csm(ew))
