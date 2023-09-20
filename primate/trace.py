# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.

from typing import * 
import numpy as np
from scipy.sparse import spmatrix, sparray
from scipy.sparse.linalg import LinearOperator

import _trace
from imate._trace_estimator import trace_estimator_utilities as te_util 
from imate._trace_estimator import trace_estimator_plot_utilities as te_plot

_builtin_matrix_functions = ["identity", "sqrt", "exp", "pow", "log", "numrank", "gaussian"]

def slq(
  A: Union[LinearOperator, spmatrix, np.ndarray],
  gram: bool = False, 
  parameters: Iterable = None,
  matrix_function: Union[str, Callable] = "identity",
  min_num_samples: int = 10,
  max_num_samples: int = 50,
  error_atol: float = None,
  error_rtol: float = 1e-2,
  confidence_level: float = 0.95,
  outlier_significance_level: float = 0.001,
  lanczos_degree: int = 20,
  lanczos_tol: int = None,
  orthogonalize: int = 0,
  num_threads: int = 0,
  verbose: bool = False,
  plot: bool = False, 
  return_info: bool = False, 
  **kwargs
):
  """Estimates the trace of a matrix function f(A) = U f(D) U^{-1} using the stochastic Lanczos quadrature (SLQ) method. 

  Parameters: 
    min_num_samples: minimum number of parameter for lanczos. 

  Reference:
    `Ubaru, S., Chen, J., and Saad, Y. (2017)
    <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_,
    Fast Estimation of :math:`\\mathrm{tr}(F(A))` Via Stochastic Lanczos
    Quadrature, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.
  """
  # assert isinstance(A, spmatrix) or isinstance(A, sparray), "A must be a sparse matrix, for now."
  attr_checks = [hasattr(A, "__matmul__"), hasattr(A, "matmul"), hasattr(A, "dot"), hasattr(A, "matvec")]
  assert any(attr_checks), "Invalid operator; must have an overloaded 'matvec' or 'matmul' method" 
  assert hasattr(A, "shape") and len(A.shape) >= 2, "Operator must be at least two dimensional."
  if gram: assert A.shape[0] == A.shape[1], "If A is a gramian matrix, it must be square!"

  ## Get the dtype; infer it if it's not available
  f_dtype = (A @ np.zeros(A.shape[1])).dtype if not hasattr(A, "dtype") else A.dtype
  i_dtype = np.int32
  
  ## Extract the machine precision for the given floating point type
  lanczos_tol = np.finfo(f_dtype).eps if lanczos_tol is None else lanczos_tol

  ## Validates operator size + initialize parameters array
  parameters = np.array([0.0], dtype=f_dtype) if parameters is None else np.fromiter(iter(parameters), dtype=f_dtype)

  # Find number of inquiries, which is the number of batches of different set
  # of parameters to produce different linear operators. These batches are
  # concatenated in the "parameters" array.
  num_inquiries = 1 #find_num_inquiries(Aop, parameters_size)

  ## Check input arguments have proper type and values
  error_atol, error_rtol = te_util.check_arguments(
    False, 1.0, min_num_samples, max_num_samples, error_atol,
    error_rtol, confidence_level, outlier_significance_level,
    lanczos_degree, lanczos_tol, orthogonalize, num_threads,
    0, verbose, plot, False
  )
  nq, ns = num_inquiries, max_num_samples                     # num queries, num samples 
  trace = np.empty((nq,), dtype=f_dtype)                      # Allocate output trace as array of size num_inquiries
  error = np.empty((nq,), dtype=f_dtype)                      # Error of computing trace within a confidence interval
  samples = np.nan * np.ones((ns, nq), dtype=f_dtype)         # Array of all Monte-Carlo samples where array trace is averaged based upon
  processed_samples_indices = np.zeros((ns,), dtype=i_dtype)  # Track the order of process of samples in rows of samples array
  num_samples_used = np.zeros((nq,), dtype=i_dtype)           # Store how many samples used for each inquiry till reaching convergence
  num_outliers = np.zeros((nq,), dtype=i_dtype)               # Number of outliers that is removed from num_samples_used in averaging
  converged = np.zeros((nq,), dtype=i_dtype)                  # Flag indicating which of the inquiries were converged below the tolerance
  alg_wall_time = np.zeros((1, ), dtype=f_dtype)              # Somewhat inaccurate measure of the total wall clock time taken 

  ## Collect the arguments processed so far 
  trace_args = (parameters, num_inquiries, 
    orthogonalize, lanczos_degree, lanczos_tol, 
    min_num_samples, max_num_samples, error_atol, error_rtol, confidence_level, outlier_significance_level, 
    num_threads, 
    trace, error, samples, 
    processed_samples_indices, num_samples_used, num_outliers, converged, alg_wall_time
  )

  ## Parameterize the matrix function and trace call
  if isinstance(matrix_function, str):
    assert matrix_function in _builtin_matrix_functions, "If given as a string, matrix_function be one of the builtin functions."
    matrix_func_id = _builtin_matrix_functions.index(matrix_function)
    method_name = "trace_" + _builtin_matrix_functions[matrix_func_id] + ("_gram" if gram else "_rect")
    inputs = [A]
    if matrix_function == "smoothstep":
      a, b = kwargs.get('a', 0.0), kwargs.get('b', 1e-6)
      inputs += [a, b]
    elif matrix_function == "numerical_rank" or matrix_function == "rank":
      threshold = kwargs.get('threshold', None)
      if threshold is None:
        from scipy.sparse.linalg import eigsh
        s_max = A.dtype.type(eigsh(A, k=1, which="LM", return_eigenvectors=False))
        threshold = s_max * np.max(A.shape) * np.finfo(A.dtype).eps
      inputs += [threshold]
    elif matrix_function == "pow":
      inputs += [kwargs.get('p', 1.0)]
    elif matrix_function == "heat":
      inputs += [kwargs.get('t', 1.0)]
    elif matrix_function == "gaussian":
      mu, sigma = kwargs.get('mu', 0.0), kwargs.get('sigma', 1.0)
      inputs += [mu, sigma]
  else:
    raise NotImplementedError("Not done yet")
  
  ## Make the actual call
  trace_f = getattr(_trace, method_name)
  trace_f(*inputs, *trace_args)

  ## If no information is required, just return the trace estimate 
  if not(return_info): 
    return trace
  else:
    ## Otherwise, collection runtime information + matrix size info (if available)
    matrix_size = A.shape[0]
    matrix_nnz = A.getnnz() if hasattr(A, "getnnz") else None
    matrix_density = A.getnnz() / np.prod(A.shape) if hasattr(A, "getnnz") else None
    sparse = None if matrix_density is None else matrix_density <= 0.50
    info = {
        'matrix':
        {
            'data_type': np.finfo(f_dtype).dtype.name.encode('utf-8'),
            'gram': False,
            'exponent': kwargs.get('p', 1.0),
            'num_inquiries': num_inquiries,
            'num_operator_parameters': 1, #Aop.get_num_parameters(),
            'parameters': parameters, 
            'size': matrix_size,               # legacy
            'sparse': sparse,
            'nnz': matrix_nnz,
            'density': matrix_density  
        },
        'error':
        {
            'absolute_error': None,
            'relative_error': None,
            'error_atol': error_atol,
            'error_rtol': error_rtol,
            'confidence_level': confidence_level,
            'outlier_significance_level': outlier_significance_level
        },
        'convergence':
        {
            'converged': converged,
            'all_converged': np.all(converged),
            'min_num_samples': min_num_samples,
            'max_num_samples': max_num_samples,
            'num_samples_used': None,
            'num_outliers': None,
            'samples': None,
            'samples_mean': None,
            'samples_processed_order': processed_samples_indices
        },
        'time':
        {
            'tot_wall_time': 0,
            'alg_wall_time': alg_wall_time,
            'cpu_proc_time': 0
        },
        'device': {
            'num_cpu_threads': num_threads,
            'num_gpu_devices': 0,
            'num_gpu_multiprocessors': 0,
            'num_gpu_threads_per_multiprocessor': 0
        },
        'solver':
        {
            'version': None,
            'lanczos_degree': lanczos_degree,
            'lanczos_tol': lanczos_tol,
            'orthogonalize': orthogonalize,
            'method': 'slq',
        }
    }

    # Fill arrays of info depending on whether output is scalar or array
    output_is_array = False if (parameters is None) or np.isscalar(parameters) else True
    if output_is_array:
      info['error']['absolute_error'] = error
      info['error']['relative_error'] = error / np.abs(trace)
      info['convergence']['converged'] = converged.astype(bool)
      info['convergence']['num_samples_used'] = num_samples_used
      info['convergence']['num_outliers'] = num_outliers
      info['convergence']['samples'] = samples
      info['convergence']['samples_mean'] = trace
    else:
      info['error']['absolute_error'] = error[0]
      info['error']['relative_error'] = error[0] / np.abs(trace[0])
      info['convergence']['converged'] = bool(converged[0])
      info['convergence']['num_samples_used'] = num_samples_used[0]
      info['convergence']['num_outliers'] = num_outliers[0]
      info['convergence']['samples'] = samples[:, 0]
      info['convergence']['samples_mean'] = trace[0]

    if verbose: te_util.print_summary(info)
    if plot: te_plot.plot_convergence(info)

    return (trace, info) if output_is_array else (trace[0], info)


# def trace_est() -> int:
#   return _trace.apply_smoothstep(1, 2)
