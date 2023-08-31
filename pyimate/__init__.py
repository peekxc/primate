
# from trace import * 
import _random_generator
import _diagonalize
import _orthogonalize
import _sparse_eigen
# from . import functions
# import _operators
# from ortho import * 

from .functions import *

# import numpy as np 
# def eigh_tridiagonal(alphas: numpy.ndarray, betas: np.ndarray, V: np.ndarray):
#   _diagonalize.eigh_tridiagonal(alphas, betas, V)
#   return V

def rademacher(n: int):
  import numpy as np
  out = np.empty(n, dtype=np.float32)
  _random_generator.rademacher(out, 1)
  return out


# Based on Numpy's usage: https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/utils.py#L75-L101
def get_include():
  """Return the directory that contains the pyimate's \\*.h header files.

  Extension modules that need to compile against pyimate should use this
  function to locate the appropriate include directory.

  Notes: 
    When using `distutils`, for example in `setup.py`:
      ```python
      import numpy as np
      ...
      Extension('extension_name', ...
              include_dirs=[np.get_include()])
      ...
      ```
    Or with `meson-python`, for example in `meson.build`:
      ```meson
      
      ...
      run_command(py, ['-c', 'import pyimate as pyim; print(pyim.get_include())', check : true).stdout().strip()
      ...
      ```
  """
  import os 
  # import pyimate
  d = os.path.join(os.path.dirname(__file__), 'include')
  return d
