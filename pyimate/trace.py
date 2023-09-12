# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.

import numpy as np
from scipy.sparse import spmatrix
from typing import * 

# import time
import _trace
from imate._trace_estimator import trace_estimator_utilities as te_util 
from imate._trace_estimator import trace_estimator_plot_utilities as te_plot

# from imate._trace_estimator.trace_estimator_utilities import get_operator_parameters, check_arguments
# from imate.trace_estimator_plot_utilities import plot_convergence
# from imate.trace_estimator_utilities import get_operator, \
#         get_operator_parameters, check_arguments, get_machine_precision, \
#         find_num_inquiries, print_summary

def slq (
  A: spmatrix,
  parameters: Iterable = None,
  matrix_function: Union[str, Callable] = "identity",
  p: int = 1.0,
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

  If 

  Reference:
    `Ubaru, S., Chen, J., and Saad, Y. (2017)
    <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_,
    Fast Estimation of :math:`\\mathrm{tr}(F(A))` Via Stochastic Lanczos
    Quadrature, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.
  """
  assert isinstance(A, spmatrix), "A must be a sparse matrix, for now."

  # Since it's just unclear how to actually run simple openmp code with python 
  num_threads = 1 if (num_threads is None or num_threads == 0) else num_threads 

  # Check operator A, and convert to a linear operator (if not already)
  # Aop = get_operator(A)
  i_dtype, f_dtype = np.int32, np.float32 
  lanczos_tol = float(np.finfo(f_dtype).eps) if lanczos_tol is None else float(lanczos_tol)

  # Validates operator size + initialize parameters array
  # parameters, parameters_size = get_operator_parameters(parameters, f_dtype)
  # parameters = parameters.astype(f_dtype)
  parameters = np.array([0.0], dtype=f_dtype)

  # Find number of inquiries, which is the number of batches of different set
  # of parameters to produce different linear operators. These batches are
  # concatenated in the "parameters" array.
  num_inquiries = 1 #find_num_inquiries(Aop, parameters_size)

  ## Check input arguments have proper type and values
  error_atol, error_rtol = te_util.check_arguments(
    False, p, min_num_samples, max_num_samples, error_atol,
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
  # alg_wall_times = np.zeros((1, ), dtype=float)

  ## Parameterize the arguments
  trace_args = (parameters, num_inquiries, 
    p, orthogonalize, lanczos_degree, lanczos_tol, 
    min_num_samples, max_num_samples, error_atol, error_rtol, confidence_level, outlier_significance_level, 
    num_threads, 
    trace, error, samples, 
    processed_samples_indices, num_samples_used, num_outliers, converged 
  )
  if isinstance(matrix_function, str):
    if matrix_function == "identity":
      alg_wall_time = _trace.trace_eigen_identity(A, *trace_args)
    elif matrix_function == "sqrt":
      alg_wall_time = _trace.trace_eigen_sqrt(A, *trace_args)
    elif matrix_function == "smoothstep":
      a, b = kwargs.get('a', 0.0), kwargs.get('b', 1e-6)
      alg_wall_time = _trace.trace_eigen_smoothstep(A, a, b, *trace_args)
    elif matrix_function == "numerical_rank" or matrix_function == "rank":
      threshold = kwargs.get('threshold', None)
      if threshold is None:
        from scipy.sparse.linalg import eigsh
        s_max = A.dtype.type(eigsh(A, k=1, which="LM", return_eigenvectors=False))
        threshold = s_max * np.max(A.shape) * np.finfo(A.dtype).eps
      alg_wall_time = _trace.trace_eigen_numrank(A, threshold, *trace_args)
    elif matrix_function == "pow":
      alg_wall_time = _trace.trace_eigen_pow(A, p, *trace_args)
    elif matrix_function == "exp":
      alg_wall_time = _trace.trace_eigen_exp(A, *trace_args)
    elif matrix_function == "log":
      alg_wall_time = _trace.trace_eigen_log(A, *trace_args)
    elif matrix_function == "inv":
      alg_wall_time = _trace.trace_eigen_inv(A, *trace_args)
    elif matrix_function == "gaussian":
      mu, sigma = kwargs.get('mu', 0.0), kwargs.get('sigma', 1.0)
      alg_wall_time = _trace.trace_eigen_gaussian(A, mu, sigma, *trace_args)
    else:
      raise ValueError(f"Unknown matrix function '{matrix_function}'")
  else:
    raise NotImplementedError("Not done yet")
    
  ## Matrix size info (if available)
  matrix_size = A.shape[0]
  matrix_nnz = A.getnnz() if hasattr(A, "getnnz") else None
  matrix_density = A.getnnz() / np.prod(A.shape) if hasattr(A, "getnnz") else None
  sparse = None if matrix_density is None else matrix_density <= 0.50

  # Dictionary of output info
  if not(return_info): return trace
  info = {
      'matrix':
      {
          'data_type': np.finfo(f_dtype).dtype.name.encode('utf-8'),
          'gram': False,
          'exponent': p,
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
