import importlib.metadata
__version__ = importlib.metadata.version("primate")

from . import plotting 
from . import ortho
from . import trace
from . import stats
from .random import rademacher, normal
from . import diagonalize
from .operator import matrix_function

# __all__ = ['rademacher', 'normal']

## Based on Numpy's usage: https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/utils.py#L75-L101
def get_include():
  """Return the directory that contains the primate's \\*.h header files.

  Extension modules that need to compile against primate should use this
  function to locate the appropriate include directory.

  Notes: 
    When using `distutils`, for example in `setup.py`:
      ```python
      import primate as pyim
      ...
      Extension('extension_name', ..., include_dirs=[pyim.get_include()])
      ...
      ```
    Or with `meson-python`, for example in `meson.build`:
      ```meson
      ...
      run_command(py, ['-c', 'import primate as pyim; print(pyim.get_include())', check : true).stdout().strip()
      ...
      ```
  """
  import os 
  d = os.path.join(os.path.dirname(__file__), 'include')
  return d


