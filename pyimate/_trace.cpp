#include <pybind11/pybind11.h>

#include <omp.h> // omp_get_num_threads
#include <cmath> // log, 
#include <_linear_operator/linear_operator.h>
#include <_definitions/types.h>
#include <_definitions/definitions.h>
#include <_trace_estimator/trace_estimator.h>
#include <_timer/timer.h>
#include "eigen_operators.h"

namespace py = pybind11;
using namespace pybind11::literals;

// For passing by reference, see: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#pass-by-reference
template< std::floating_point F, Operator Matrix, std::invocable< F > Func > 
F trace_estimator_slq_py(
  const Matrix* matrix, 
  Func&& matrix_function, 
  const py::array_t< F >& parameters, 
  const size_t num_inqueries,
  const F exponent, 
  const int orthogonalize, 
  const size_t lanczos_degree, 
  const F lanczos_tol, 
  const size_t min_num_samples, 
  const size_t max_num_samples, 
  const F error_atol, 
  const F error_rtol,
  const F confidence, 
  const F outlier,
  const int num_threads, 
  py::array_t< F >& trace,
  py::array_t< F >& error,
  py::array_t< F >& samples,
  py::array_t< int >& processed_samples_indices,
  py::array_t< int >& num_samples_used,
  py::array_t< int >& num_outliers,
  py::array_t< int >& converged
){
  // Set the number of threads based on user input
  const int num_threads_ = num_threads < 1 ? omp_get_max_threads() : std::max(num_threads, 1);

  // if (gramian){ throw std::invalid_argument("Gramian not available yet."); }
  const F* params = parameters.data(); // static_cast< float* >(parameters.request().ptr); // 
  F* trace_out = static_cast< F* >(trace.request().ptr); 
  F* error_out = static_cast< F* >(error.request().ptr); 
  int* processed_samples_indices_out = static_cast< int* >(processed_samples_indices.request().ptr); 
  int* num_samples_used_out = static_cast< int* >(num_samples_used.request().ptr); 
  int* num_outliers_out = static_cast< int* >(num_outliers.request().ptr); 
  int* converged_out = static_cast< int* >(converged.request().ptr); 
  F alg_wall_time = 0.0; 

  // Convert samples 2d array 
  py::buffer_info samples_buffer = samples.request();
  F* samples_ptr = static_cast< F* >(samples_buffer.ptr);   
  ssize_t s_rows = samples.shape(0);
  ssize_t s_cols = samples.shape(1);
  F** samples_out = new F*[s_rows];
  for (ssize_t i = 0; i < s_rows; ++i) {
    samples_out[i] = samples_ptr + i * s_cols;
  }
  
  // Call the trace estimators
  {
    // py::gil_scoped_release gil_release; // this is safe, but doesn't appear to be necessary
    trace_estimator< false, F >(
      matrix, params, num_inqueries, matrix_function, 
      exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, 
      error_atol, error_rtol, confidence, outlier, 
      num_threads_, 
      trace_out, error_out, samples_out, processed_samples_indices_out, num_samples_used_out, num_outliers_out, converged_out,
      alg_wall_time
    );
  }

  delete[] samples_out;
  return alg_wall_time; 
}

#define TRACE_PARAMS \
  const py::array_t< F >& parameters, const size_t num_inqueries, \
  const F exponent, const int orthogonalize, const size_t lanczos_degree, const F lanczos_tol, const size_t min_num_samples, const size_t max_num_samples, \
  const F error_atol,  const F error_rtol, const F confidence, const F outlier, \
  const int num_threads, \
  py::array_t< F >& trace, py::array_t< F >& error, py::array_t< F >& samples, \
  py::array_t< int >& processed_samples_indices, py::array_t< int >& num_samples_used, py::array_t< int >& num_outliers, py::array_t< int >& converged

#define TRACE_ARGS \
  parameters, num_inqueries, \
  exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, \
  error_atol, error_rtol, confidence, outlier, \
  num_threads, \
  trace, error, samples, \
  processed_samples_indices, num_samples_used, num_outliers, converged

template< std::floating_point F = float, std::invocable< F > MatrixFunc >
auto trace_eigen(const Eigen::SparseMatrix< F >* A, MatrixFunc&& f, TRACE_PARAMS) -> F {
  auto B = Eigen::SparseMatrix< F >(A->rows(), A->cols());
  B.setIdentity();
  const auto lo = SparseEigenAffineOperator(*A, B, 0.0);
  F alg_time = trace_estimator_slq_py< F >(&lo, f, TRACE_ARGS);
  return alg_time; 
}

// See: https://stackoverflow.com/questions/67071795/calling-parallel-c-code-in-python-using-pybind11
template< std::floating_point F >
void eigen_trace(py::module &m){
  m.def("trace_eigen_identity", [](const Eigen::SparseMatrix< F >* A, TRACE_PARAMS){
    const auto f = std::identity(); //[](F eigenvalue) -> F { return eigenvalue; }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_smoothstep", [](const Eigen::SparseMatrix< F >* A, const F a, const F b, TRACE_PARAMS){
    const F d = (b-a);
    const auto f = [a, d](F eigenvalue) -> F { 
      return std::min(std::max((eigenvalue-a)/d, F(0.0)), F(1.0)); 
    }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_sqrt", [](const Eigen::SparseMatrix< F >* A, TRACE_PARAMS){
    const auto f = [](F eigenvalue) -> F {  return std::sqrt(eigenvalue); }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_inv", [](const Eigen::SparseMatrix< F >* A, TRACE_PARAMS){
    const auto f = [](F eigenvalue) -> F {  return (F(1.0)/eigenvalue); }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_log", [](const Eigen::SparseMatrix< F >* A, TRACE_PARAMS){
    const auto f = [](F eigenvalue) -> F {  return std::log(eigenvalue); }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_exp", [](const Eigen::SparseMatrix< F >* A, TRACE_PARAMS){
    const auto f = [](F eigenvalue) -> F {  return std::exp(eigenvalue); }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_pow", [](const Eigen::SparseMatrix< F >* A, const F p, TRACE_PARAMS){
    const auto f = [p](F eigenvalue) -> F {  return std::pow(eigenvalue, p); }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_gaussian", [](const Eigen::SparseMatrix< F >* A, const F mu, const F sigma, TRACE_PARAMS){
    const auto f = [mu, sigma](F eigenvalue) -> F {  
      auto x = (eigenvalue - mu) / sigma;
      return (0.5 * M_SQRT1_2 * M_2_SQRTPI / sigma) * exp(-0.5 * x * x); 
    }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
  m.def("trace_eigen_numrank", [](const Eigen::SparseMatrix< F >* A, const F threshold, TRACE_PARAMS){
    const auto f = [threshold](F eigenvalue) -> F {  
      return eigenvalue > threshold ? F(1.0) : F(0.0);
    }; 
    return trace_eigen< F >(A, f, TRACE_ARGS);
  });
}

// Turns out using py::call_guard<py::gil_scoped_release>() just causes everthing to crash immediately
PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";
  eigen_trace< float >(m);
  eigen_trace< double >(m);
  eigen_trace< long double >(m);
};