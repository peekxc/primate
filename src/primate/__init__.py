import importlib.metadata
# from . import _lanczos_pythran
# from . import lanczos
# from .lanczos import lanczos

# from .lanczos import lanczos, rayleigh_ritz
# from .quadrature import lanczos_quadrature
# from .operators import MatrixFunction
# from .stochastic import isotropic
# from .estimators import hutch

__version__ = importlib.metadata.version("scikit-primate")


## Based on Numpy's usage: https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/utils.py#L75-L101
def get_include():
	"""Return the directory that contains the primate's .h header files.

	Extension modules that need to compile against primate should use this
	function to locate the appropriate include directory.

	Notes:
	  When using `distutils`, for example in `setup.py`:
	    ```python
	    import primate
	    ...
	    Extension('extension_name', ..., include_dirs=[primate.get_include()])
	    ...
	    ```
	  Or with `meson-python`, for example in `meson.build`:
	    ```meson
	    ...
	    run_command(py, ['-c', 'import primate; print(primate.get_include())', check : true).stdout().strip()
	    ...
	    ```
	"""
	import os

	d = os.path.join(os.path.dirname(__file__), "include")
	return d
