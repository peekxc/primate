from typing import * 
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

## Package imports
from .random import _engine_prefixes, _engines
import _lanczos

## Natively support matrix functions
_builtin_matrix_functions = ["identity", "sqrt", "exp", "pow", "log", "numrank", "smoothstep", "gaussian"]

def sl_trace (
  A: Union[LinearOperator, np.ndarray],
  fun: Union[str, Callable] = "identity", 
  maxiter: int = 200,
  deg: int = 20,
  atol: float = None,
  rtol: float = None,
  stop: str = ["confidence", "change"],
  orth: int = 0,
  confidence: float = 0.95,
  pdf: str = "rademacher",
  rng: str = "lcg",
  seed: int = -1,
  num_threads: int = 0,
  verbose: bool = False,
  info: bool = False, 
  plot: bool = False,
  **kwargs
) -> Union[float, tuple]:
  """Estimates the trace of a matrix function $f(A)$ using stochastic Lanczos quadrature (SLQ). 

  Parameters
  ----------
  A : ndarray, sparse matrix, or LinearOperator
      real, square, symmetric operator given as a ndarray, a sparse matrix, or a LinearOperator. 
  fun : str or Callable, default="identity"
      real-valued function defined on the spectrum of A. 
  maxiter : int, default = 10
      Maximum number of random vectors to sample for the trace estimate. 
  deg  : int, default = 20
      Degree of the quadrature approximation.     
  atol : float, default = None
      Absolute tolerance to signal convergence for early-stopping. See details.  
  rtol : float, default = 1e-2
      Relative tolerance to signal convergence for early-stopping. See details.
  orth: int, default = 0,
      Additional number of Lanczos vectors to orthogonalize against when building the Krylov basis.   
  confidence : float, default = 0.95
      Confidence level to use with rule  
  pdf : { 'rademacher', 'normal' }, default = "rademacher"
      Choice of zero-centered distribution to sample random vectors from.
  rng : str, default = "pcg"
      Random number generator to use. Defaults to PCG64 generator. 
  seed : int, default = -1
      Seed to initialize the entropy source. Use non-negative integers for reproducibility.
  num_threads: int, default = 0
      Number of threads to use to parallelize the computation. Use values <= 0 to maximize the number of threads. 
  plot : bool, default = False
      If true, plots the samples of the trace estimate along with their convergence characteristics. 
  info: bool, default = False
      If True, returns a dictionary containing all relevant information about the computation. 
  kwargs : dict, optional 
      additional key-values to parameterize the chosen function 'fun'.
      
  Returns
  -------
  trace_estimate : float 
      Estimate of the trace of the matrix function $f(A)$.
  info : dict, optional 
      If 'info = True', additional information about the computation. 

  See Also
  --------
  lanczos : the lanczos algorithm. 

  Reference
  ---------
    [1] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. 
    SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
  """
  attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
  assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method" 
  assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
  assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"
  
  ## Choose the random number engine 
  assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
  engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)

  ## Choose the distribution to sample random vectors from 
  assert pdf in [ "rademacher", "normal"], f"Invalid distribution '{pdf}'; Must be one of 'rademacher' or 'normal'."
  distr_id = ["rademacher", "normal"].index(pdf)

  ## Choose the stopping criteria
  if stop == ["confidence", "change"] or stop == "confidence":
    use_clt: bool = True 
  elif stop == "change":
    use_clt: bool = False
  else: 
    raise ValueError(f"Invalid convergence criteria '{str(stop)}' supplied.") 

  ## Get the dtype; infer it if it's not available
  f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
  i_dtype = np.int32
  assert f_dtype.type == np.float32 or f_dtype.type == np.float64, "Only 32- or 64-bit floating point numbers are supported."

  ## Extract the machine precision for the given floating point type
  lanczos_rtol = np.finfo(f_dtype).eps  # if lanczos_rtol is None else f_dtype.type(lanczos_rtol)

  ## Check input arguments have proper type and values
  error_atol = f_dtype.type(0.0) if atol is None else f_dtype.type(atol)
  error_rtol = f_dtype.type(rtol)

  ## Argument checking
  nv = int(maxiter)                                          # Number of random vectors to generate
  seed = int(seed)                                      # Seed should be an integer
  orth = int(orth)                    # Number of additional vectors should be an integer
  # alg_wall_time = f_dtype.type(0.0)                     # Total time spent by the algorithm
  deg = max(deg, 2)               # Should be at least two 
  ncv = int(deg + orth)             # Number of Lanczos vectors to keep in memory
  atol = 0.0 if atol is None else float(atol)           # Early stopping upper bound on confidence interval 
  rtol = 0.0 if rtol is None else float(rtol)           # Early stopper relative standard error bound
  num_threads = 0 if num_threads < 0 else int(num_threads)
  
  ## Adjust tolerance for the quadrature estimates
  atol /= A.shape[1]  

  ## Parameterize the matrix function and trace call
  if isinstance(fun, str):
    assert fun in _builtin_matrix_functions, "If given as a string, matrix_function be one of the builtin functions."
    kwargs["function"] = fun # _builtin_matrix_functions.index(matrix_function)
  elif isinstance(fun, Callable):
    kwargs["function"] = "generic"
    kwargs["matrix_func"] = fun
  else: 
    raise ValueError(f"Invalid matrix function type '{type(fun)}'")
  
  ## Collect the arguments processed so far 
  sl_trace_args = (
    nv, distr_id, engine_id, seed, 
    deg, 0.0, orth, ncv, 
    atol, rtol, 
    num_threads, 
    use_clt
  )

  ## Make the actual call
  estimates = _lanczos.stochastic_trace(A, *sl_trace_args, **kwargs)
  estimates *= A.shape[1]

  ## Plot the trace estimates 
  if plot: 
    from bokeh.plotting import show
    from .plotting import figure_trace
    show(figure_trace(estimates))

  ## If requested, create the info dictionary; o/w just return the point-estimate
  trace_estimate = np.mean(estimates)
  if not info: return trace_estimate
  info = {
    'estimate' : trace_estimate,
    'samples' : estimates
  }
  return trace_estimate, info

  ## If no information is required, just return the trace estimate 
  # if not(return_info) and not(plot) and not(verbose): 
  #   return trace
  # else:
  #   ## Otherwise, collection runtime information + matrix size info (if available)
  #   matrix_size = A.shape[0]
  #   matrix_nnz = A.getnnz() if hasattr(A, "getnnz") else None
  #   matrix_density = A.getnnz() / np.prod(A.shape) if hasattr(A, "getnnz") else None
  #   sparse = None if matrix_density is None else matrix_density <= 0.50
  #   info = { }
  #   info['error'] = dict(
  #     absolute_error=None, relative_error=None, error_atol=error_atol, error_rtol=error_rtol, 
  #     confidence_level=confidence_level, outlier_significance_level=outlier_significance_level
  #   )
  #   info['matrix'] = dict(
  #     data_type = np.finfo(f_dtype).dtype.name.encode('utf-8'), gram=False, exponent=kwargs.get('p', 1.0),
  #     num_inquiries=num_inquiries, num_operator_parameters=1, parameters=parameters,
  #     size=matrix_size, sparse=sparse, nnz=matrix_nnz, density=matrix_density
  #   ),
  #   info['error'] = {
  #     'absolute_error': None,
  #     'relative_error': None,
  #     'error_atol': error_atol,
  #     'error_rtol': error_rtol,
  #     'confidence_level': confidence_level,
  #     'outlier_significance_level': outlier_significance_level
  #   }
  #   info['convergence'] = {
  #     'converged': converged,
  #     'all_converged': np.all(converged),
  #     'min_num_samples': min_num_samples,
  #     'max_num_samples': max_num_samples,
  #     'num_samples_used': None,
  #     'num_outliers': None,
  #     'samples': None,
  #     'samples_mean': None,
  #     'samples_processed_order': processed_samples_indices
  #   }
  #   info['time'] = {
  #     'tot_wall_time': 0,
  #     'alg_wall_time': alg_wall_time,
  #     'cpu_proc_time': 0
  #   }
  #   info['device'] = {
  #     'num_cpu_threads': num_threads,
  #     'num_gpu_devices': 0,
  #     'num_gpu_multiprocessors': 0,
  #     'num_gpu_threads_per_multiprocessor': 0
  #   }
  #   info['solver'] = {
  #     'version': None,
  #     'lanczos_degree': lanczos_degree,
  #     'lanczos_tol': lanczos_tol,
  #     'orthogonalize': orthogonalize,
  #     'method': 'slq',
  #   }

  #   # Fill arrays of info depending on whether output is scalar or array
  #   output_is_array = False if (parameters is None) or np.isscalar(parameters) else True
  #   if output_is_array:
  #     info['error']['absolute_error'] = error
  #     info['error']['relative_error'] = error / np.abs(trace)
  #     info['convergence']['converged'] = converged.astype(bool)
  #     info['convergence']['num_samples_used'] = num_samples_used
  #     info['convergence']['num_outliers'] = num_outliers
  #     info['convergence']['samples'] = samples
  #     info['convergence']['samples_mean'] = trace
  #   else:
  #     info['error']['absolute_error'] = error[0]
  #     info['error']['relative_error'] = error[0] / np.abs(trace[0])
  #     info['convergence']['converged'] = bool(converged[0])
  #     info['convergence']['num_samples_used'] = num_samples_used[0]
  #     info['convergence']['num_outliers'] = num_outliers[0]
  #     info['convergence']['samples'] = samples[:, 0]
  #     info['convergence']['samples_mean'] = trace[0]

  #   # if verbose: te_util.print_summary(info)
  #   # if plot: te_plot.plot_convergence(info)
  #   if plot: 
  #     from .plotting import plot_trace
  #     plot_trace(info)
    
  #   return (trace, info) if output_is_array else (trace[0], info)


def sl_gauss(
  A: Union[LinearOperator, np.ndarray],
  n: int = 150,
  deg: int = 20,
  pdf: str = "rademacher",
  rng: str = "pcg",
  seed: int = -1,
  orth: int = 0,
  num_threads: int = 0
) -> np.ndarray:
  """Gaussian quadrature.
  
  Computes the sample nodes and weights for the degree k orthogonal polynomial approximating the cumulative spectral measure mu(t) of A. 
  These nodes/weights represent the quadrature rule for the Riemann-Stieltjes integral w.r.t. mu(t).  
  """
  attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
  assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method" 
  assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
  assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"
  
  ## Choose the random number engine 
  assert rng in _engine_prefixes or rng in _engines, f"Invalid pseudo random number engine supplied '{str(rng)}'"
  engine_id = _engine_prefixes.index(rng) if rng in _engine_prefixes else _engines.index(rng)

  ## Choose the distribution to sample random vectors from 
  assert pdf in [ "rademacher", "normal"], f"Invalid distribution '{pdf}'; Must be one of 'rademacher' or 'normal'."
  distr_id = ["rademacher", "normal"].index(pdf)

  ## Get the dtype; infer it if it's not available
  f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
  i_dtype = np.int32
  assert f_dtype.type == np.float32 or f_dtype.type == np.float64, "Only 32- or 64-bit floating point numbers are supported."

  ## Extract the machine precision for the given floating point type
  lanczos_rtol = np.finfo(f_dtype).eps  # if lanczos_rtol is None else f_dtype.type(lanczos_rtol)

  ## Argument checking
  m = A.shape[1]                                        # Dimension of the space
  nv = int(n)                                          # Number of random vectors to generate
  seed = int(seed)                                      # Seed should be an integer
  deg = max(deg, 2)               # Must be at least 2 
  orth = m - 1 if orth < 0 else min(m - 1, orth)                   # Number of additional vectors should be an integer
  ncv = max(int(deg + orth), m)             # Number of Lanczos vectors to keep in memory
  num_threads = int(num_threads)                        # should be integer; if <= 0 will trigger max threads on C++ side

  ## Collect the arguments processed so far 
  sl_quad_args = (
    nv, distr_id, engine_id, seed, 
    deg, lanczos_rtol, orth, ncv, 
    num_threads
  )

  ## Make the actual call
  quad_nw = _lanczos.stochastic_quadrature(A, *sl_quad_args)
  return quad_nw



# def trace_est() -> int:
#   return _trace.apply_smoothstep(1, 2)


# def slq (
#   A: Union[LinearOperator, np.ndarray],
#   matrix_function: Union[str, Callable] = "identity", 
#   parameters: Iterable = None,
#   min_num_samples: int = 150,
#   max_num_samples: int = 250,
#   error_atol: float = None,
#   error_rtol: float = 1e-2,
#   confidence_level: float = 0.95,
#   outlier_significance_level: float = 0.001,
#   distribution: str = "rademacher",
#   rng_engine: str = "pcg",
#   seed: int = -1,
#   lanczos_degree: int = 20,
#   lanczos_tol: int = None,
#   orthogonalize: int = 0,
#   num_threads: int = 0,
#   verbose: bool = False,
#   plot: bool = False, 
#   return_info: bool = False, 
#   **kwargs
# ):
#   """Estimates the trace of a matrix function $f(A)$ using stochastic Lanczos quadrature (SLQ). 

#   Parameters
#   ----------
#   A : ndarray, sparse matrix, or LinearOperator
#       real, square, symmetric operator given as a ndarray, a sparse matrix, or a LinearOperator. 
#   matrix_function : str or Callable, default="identity"
#       float-valued function defined on the spectrum of A. 
#   parameters : Iterable, default = None
#       translates 't' for the affine operator A + t*B (see details). 
#   min_num_samples : int, default = 10
#       Minimum number of random vectors to sample for the trace estimate. 
#   max_num_samples : int, default = 50
#       Maximum number of random vectors to sample for the trace estimate. 
#   error_atol : float, default = None
#       Absolute tolerance governing convergence. 
#   error_rtol : float, default = 1e-2
#       Relative tolerance governing convergence. 
#   confidence_level : float, default = 0.95
#       Confidence level before converging. 
#   outlier_significance_level: float, default = 0.001
#       Outliers to ignore in trace estimation. 
#   distribution : { 'rademacher', 'normal' }, default = "rademacher"
#       zero-centered distribution to sample random vectors from.
#   rng_engine : str, default = "pcg"
#       Random number engine to use.
#   seed : int, default = -1
#       Seed to initialize the entropy source. Use non-negative integers for reproducibility.
#   lanczos_degree  : int, default = 20
#       Degree of the quadrature approximation. 
#   lanczos_tol : int, default = None
#       Acceptable residual tolerance to prematurely stop the lanczos iteration. 
#   orthogonalize: int, default = 0,
#       Number of lanczos vectors to orthogonalize the Krylov basis. 
#   num_threads: int, default = 0
#       Number of threads to use in the trace estimates. 
#   plot : bool, default = False
#       If true, plots the samples of the trace estimate along with their convergence characteristics. 
#   return_info : bool, default = False
#       whether to return a dictionary containing all the relevent details of the computation.
#   **kwargs : dict, optional 
#       additional key-values to parameterize the chosen 'matrix_function'.
      
#   Returns
#   -------
#   trace_estimate : float 
#       Estimate of the trace of the matrix function $f(A)$.
#   info : dict, optional 
#       If 'return_info = True', additional information about the computation. 

#   See Also
#   --------
#   lanczos : the lanczos algorithm. 

#   Reference
#   ---------
#     .. [1] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. 
#     SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
#   """
#   # assert isinstance(A, spmatrix) or isinstance(A, sparray), "A must be a sparse matrix, for now."
#   attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
#   assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method" 
#   assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
#   assert A.shape[0] == A.shape[1], "This function only works with square, symmetric matrices!"
  
#   ## Choose the random number engine 
#   assert rng_engine in _engine_prefixes or rng_engine in _engines, f"Invalid pseudo random number engine supplied '{str(rng_engine)}'"
#   engine_id = _engine_prefixes.index(rng_engine) if rng_engine in _engine_prefixes else _engines.index(rng_engine)

#   ## Choose the distribution to sample random vectors from 
#   assert distribution in [ "rademacher", "normal"], f"Invalid distribution '{distribution}'; Must be one of 'rademacher' or 'normal'."
#   distr_id = ["rademacher", "normal"].index(distribution)

#   ## Get the dtype; infer it if it's not available
#   f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
#   i_dtype = np.int32
#   assert f_dtype.type == np.float32 or f_dtype.type == np.float64, "Only 32- or 64-bit floating point numbers are supported."

#   ## Extract the machine precision for the given floating point type
#   lanczos_tol = np.finfo(f_dtype).eps if lanczos_tol is None else f_dtype.type(lanczos_tol)

#   ## Validates operator size + initialize parameters array
#   parameters = np.array([0.0], dtype=f_dtype) if parameters is None else np.fromiter(iter(parameters), dtype=f_dtype)

#   # Find number of inquiries, which is the number of batches of different set
#   # of parameters to produce different linear operators. These batches are
#   # concatenated in the "parameters" array.
#   num_inquiries = 1 #find_num_inquiries(Aop, parameters_size)

#   ## Check input arguments have proper type and values
#   error_atol = f_dtype.type(1e-2) if error_atol is None else f_dtype.type(error_atol)
#   error_rtol = f_dtype.type(error_rtol)
#   # te_util.check_arguments(
#   #   gram, 1.0, min_num_samples, max_num_samples, error_atol,
#   #   error_rtol, confidence_level, outlier_significance_level,
#   #   lanczos_degree, lanczos_tol, orthogonalize, num_threads,
#   #   0, verbose, plot, False
#   # )
#   nq, ns = num_inquiries, max_num_samples                     # num queries, num samples 
#   trace = np.empty((nq,), dtype=f_dtype)                      # Allocate output trace as array of size num_inquiries
#   error = np.empty((nq,), dtype=f_dtype)                      # Error of computing trace within a confidence interval
#   samples = np.nan * np.ones((ns, nq), dtype=f_dtype)         # Array of all Monte-Carlo samples where array trace is averaged based upon
#   processed_samples_indices = np.zeros((ns,), dtype=i_dtype)  # Track the order of process of samples in rows of samples array
#   num_samples_used = np.zeros((nq,), dtype=i_dtype)           # Store how many samples used for each inquiry till reaching convergence
#   num_outliers = np.zeros((nq,), dtype=i_dtype)               # Number of outliers that is removed from num_samples_used in averaging
#   converged = np.zeros((nq,), dtype=i_dtype)                  # Flag indicating which of the inquiries were converged below the tolerance
#   alg_wall_time = f_dtype.type(0.0)                     # Somewhat inaccurate measure of the total wall clock time taken 
#   lanczos_degree = max(lanczos_degree, 2)               # should be at least two 
  
#   ## Collect the arguments processed so far 
#   trace_args = (parameters, num_inquiries, 
#     orthogonalize, lanczos_degree, lanczos_tol, 
#     min_num_samples, max_num_samples, error_atol, error_rtol, confidence_level, outlier_significance_level,
#     distr_id, engine_id, seed, 
#     num_threads, 
#     trace, error, samples, 
#     processed_samples_indices, num_samples_used, num_outliers, converged, alg_wall_time
#   )

#   ## Parameterize the matrix function and trace call
#   if isinstance(matrix_function, str):
#     assert matrix_function in _builtin_matrix_functions, "If given as a string, matrix_function be one of the builtin functions."
#     #matrix_func_id = _builtin_matrix_functions.index(matrix_function)
#     kwargs["function"] = matrix_function
#   elif isinstance(matrix_function, Callable):
#     kwargs["function"] = "generic"
#     kwargs["matrix_func"] = matrix_function
#   else: 
#     raise ValueError(f"Invalid matrix function type '{type(matrix_function)}'")
  
#   ## Make the actual call
#   _trace.trace_slq(A, *trace_args, **kwargs)

#   ## If no information is required, just return the trace estimate 
#   if not(return_info) and not(plot) and not(verbose): 
#     return trace
#   else:
#     ## Otherwise, collection runtime information + matrix size info (if available)
#     matrix_size = A.shape[0]
#     matrix_nnz = A.getnnz() if hasattr(A, "getnnz") else None
#     matrix_density = A.getnnz() / np.prod(A.shape) if hasattr(A, "getnnz") else None
#     sparse = None if matrix_density is None else matrix_density <= 0.50
#     info = { }
#     info['error'] = dict(
#       absolute_error=None, relative_error=None, error_atol=error_atol, error_rtol=error_rtol, 
#       confidence_level=confidence_level, outlier_significance_level=outlier_significance_level
#     )
#     info['matrix'] = dict(
#       data_type = np.finfo(f_dtype).dtype.name.encode('utf-8'), gram=False, exponent=kwargs.get('p', 1.0),
#       num_inquiries=num_inquiries, num_operator_parameters=1, parameters=parameters,
#       size=matrix_size, sparse=sparse, nnz=matrix_nnz, density=matrix_density
#     ),
#     info['error'] = {
#       'absolute_error': None,
#       'relative_error': None,
#       'error_atol': error_atol,
#       'error_rtol': error_rtol,
#       'confidence_level': confidence_level,
#       'outlier_significance_level': outlier_significance_level
#     }
#     info['convergence'] = {
#       'converged': converged,
#       'all_converged': np.all(converged),
#       'min_num_samples': min_num_samples,
#       'max_num_samples': max_num_samples,
#       'num_samples_used': None,
#       'num_outliers': None,
#       'samples': None,
#       'samples_mean': None,
#       'samples_processed_order': processed_samples_indices
#     }
#     info['time'] = {
#       'tot_wall_time': 0,
#       'alg_wall_time': alg_wall_time,
#       'cpu_proc_time': 0
#     }
#     info['device'] = {
#       'num_cpu_threads': num_threads,
#       'num_gpu_devices': 0,
#       'num_gpu_multiprocessors': 0,
#       'num_gpu_threads_per_multiprocessor': 0
#     }
#     info['solver'] = {
#       'version': None,
#       'lanczos_degree': lanczos_degree,
#       'lanczos_tol': lanczos_tol,
#       'orthogonalize': orthogonalize,
#       'method': 'slq',
#     }

#     # Fill arrays of info depending on whether output is scalar or array
#     output_is_array = False if (parameters is None) or np.isscalar(parameters) else True
#     if output_is_array:
#       info['error']['absolute_error'] = error
#       info['error']['relative_error'] = error / np.abs(trace)
#       info['convergence']['converged'] = converged.astype(bool)
#       info['convergence']['num_samples_used'] = num_samples_used
#       info['convergence']['num_outliers'] = num_outliers
#       info['convergence']['samples'] = samples
#       info['convergence']['samples_mean'] = trace
#     else:
#       info['error']['absolute_error'] = error[0]
#       info['error']['relative_error'] = error[0] / np.abs(trace[0])
#       info['convergence']['converged'] = bool(converged[0])
#       info['convergence']['num_samples_used'] = num_samples_used[0]
#       info['convergence']['num_outliers'] = num_outliers[0]
#       info['convergence']['samples'] = samples[:, 0]
#       info['convergence']['samples_mean'] = trace[0]

#     # if verbose: te_util.print_summary(info)
#     # if plot: te_plot.plot_convergence(info)
#     if plot: 
#       from .plotting import plot_trace
#       plot_trace(info)
    
#     return (trace, info) if output_is_array else (trace[0], info)


# # def trace_est() -> int:
# #   return _trace.apply_smoothstep(1, 2)
