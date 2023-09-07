#include <pybind11/pybind11.h>

#include <_linear_operator/linear_operator.h>
#include <_definitions/types.h>
#include <_definitions/definitions.h>
#include <_trace_estimator/trace_estimator.h>
#include "eigen_operators.h"

namespace py = pybind11;
using py_arr_f = py::array_t< float >;
using namespace pybind11::literals;

// For passing by reference, see: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#pass-by-reference
template< std::floating_point DataType, Operator Matrix, std::invocable< DataType > Func > 
float trace_estimator_slq_py(
  const Matrix* matrix, 
  Func&& matrix_function, 
  const py_arr_f& parameters, 
  const size_t num_inqueries,
  const float exponent, 
  const int orthogonalize, 
  const size_t lanczos_degree, 
  const float lanczos_tol, 
  const size_t min_num_samples, 
  const size_t max_num_samples, 
  const float error_atol, 
  const float error_rtol,
  const float confidence, 
  const float outlier,
  const int num_threads, 
  py_arr_f& trace,
  py_arr_f& error,
  py_arr_f& samples,
  py::array_t< int >& processed_samples_indices,
  py::array_t< int >& num_samples_used,
  py::array_t< int >& num_outliers,
  py::array_t< int >& converged
){
  // Set the number of threads based on user input
  int num_threads_ = num_threads < 1 ? omp_get_max_threads() : std::max(num_threads, 1);

  // if (gramian){ throw std::invalid_argument("Gramian not available yet."); }
  float* params = static_cast< float* >(parameters.request().ptr);
  float* trace_out = static_cast< float* >(trace.request().ptr); 
  float* error_out = static_cast< float* >(error.request().ptr); 
  int* processed_samples_indices_out = static_cast< int* >(processed_samples_indices.request().ptr); 
  int* num_samples_used_out = static_cast< int* >(num_samples_used.request().ptr); 
  int* num_outliers_out = static_cast< int* >(num_outliers.request().ptr); 
  int* converged_out = static_cast< int* >(converged.request().ptr); 
  float alg_wall_time = 0.0; 

  // Convert samples 2d array 
  py::buffer_info samples_buffer = samples.request();
  float* samples_ptr = static_cast< float* >(samples_buffer.ptr);   
  ssize_t s_rows = samples.shape(0);
  ssize_t s_cols = samples.shape(1);
  float** samples_out = new float*[s_rows];
  for (ssize_t i = 0; i < s_rows; ++i) {
    samples_out[i] = samples_ptr + i * s_cols;
  }

  trace_estimator< false, float >(
    matrix, params, num_inqueries, matrix_function, 
    exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, 
    error_atol, error_rtol, confidence, outlier, 
    num_threads_, 
    trace_out, error_out, samples_out, processed_samples_indices_out, num_samples_used_out, num_outliers_out, converged_out,
    alg_wall_time
  );

  delete[] samples_out;
  return alg_wall_time; 
}

#define TRACE_ARG_DEFS \
  const py_arr_f& parameters, const size_t num_inqueries, \
  const float exponent, const int orthogonalize, const size_t lanczos_degree, const float lanczos_tol, const size_t min_num_samples, const size_t max_num_samples, \
  const float error_atol,  const float error_rtol, const float confidence, const float outlier, \
  const int num_threads, \
  py_arr_f& trace, py_arr_f& error, py_arr_f& samples, \
  py::array_t< int >& processed_samples_indices, py::array_t< int >& num_samples_used, py::array_t< int >& num_outliers, py::array_t< int >& converged

#define TRACE_ARGS \
  parameters, num_inqueries, \
  exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, \
  error_atol, error_rtol, confidence, outlier, \
  num_threads, \
  trace, error, samples, \
  processed_samples_indices, num_samples_used, num_outliers, converged


float trace_eigen_identity(const Eigen::SparseMatrix< float >* A, TRACE_ARG_DEFS) {
  auto B = Eigen::SparseMatrix< float >(A->rows(), A->cols());
  B.setIdentity();
  const auto lo = SparseEigenAffineOperator(*A, B, 0.0);
  const auto matrix_function = [](float eigenvalue) -> float { return eigenvalue; }; 
  float alg_time = trace_estimator_slq_py< float >(&lo, matrix_function, TRACE_ARGS);
  return alg_time; 
}

float trace_eigen_sqrt(const Eigen::SparseMatrix< float >* A, TRACE_ARG_DEFS) {
  auto B = Eigen::SparseMatrix< float >(A->rows(), A->cols());
  B.setIdentity();
  const auto lo = SparseEigenAffineOperator(*A, B, 0.0);
  const auto matrix_function = [](float eigenvalue) -> float { return std::sqrt(eigenvalue); }; 
  float alg_time = trace_estimator_slq_py< float >(&lo, matrix_function, TRACE_ARGS);
  return alg_time; 
}

float trace_eigen_smoothstep(const Eigen::SparseMatrix< float >* A, const float a, const float b, TRACE_ARG_DEFS) {
  float d = (b-a);
  const auto matrix_function = [a, d](float eigenvalue) -> float { 
    return std::min(std::max((eigenvalue-a)/d, 0.0f), 1.0f); 
  }; 
  auto B = Eigen::SparseMatrix< float >(A->rows(), A->cols());
  B.setIdentity();
  const auto lo = SparseEigenAffineOperator(*A, B, 0.0);
  float alg_time = trace_estimator_slq_py< float >(&lo, matrix_function, TRACE_ARGS);
  return alg_time; 
}



float trace_estimator_slq_py2(
  const Eigen::SparseMatrix< float >* matrix, 
  const py_arr_f& parameters, 
  const size_t num_inqueries,
  const float exponent, 
  const int orthogonalize, 
  const size_t lanczos_degree, 
  const float lanczos_tol, 
  const size_t min_num_samples, 
  const size_t max_num_samples, 
  const float error_atol, 
  const float error_rtol,
  const float confidence, 
  const float outlier,
  const int num_threads, 
  py_arr_f& trace,
  py_arr_f& error,
  py_arr_f& samples,
  py::array_t< int >& processed_samples_indices,
  py::array_t< int >& num_samples_used,
  py::array_t< int >& num_outliers,
  py::array_t< int >& converged
){
  int num_threads_ = num_threads;
  if (num_threads_ < 1) {
    num_threads_ = omp_get_max_threads();
  }
  num_threads_ = std::max(num_threads_, 1);

  // if (gramian){ throw std::invalid_argument("Gramian not available yet."); }
  float* params = static_cast< float* >(parameters.request().ptr);
  auto matrix_function = [](float eigenvalue) -> float { return eigenvalue; }; 
  float* trace_out = static_cast< float* >(trace.request().ptr); 
  float* error_out = static_cast< float* >(error.request().ptr); 
  int* processed_samples_indices_out = static_cast< int* >(processed_samples_indices.request().ptr); 
  int* num_samples_used_out = static_cast< int* >(num_samples_used.request().ptr); 
  int* num_outliers_out = static_cast< int* >(num_outliers.request().ptr); 
  int* converged_out = static_cast< int* >(converged.request().ptr); 
  float alg_wall_time = 0.0; 

  // Convert samples 2d array 
  py::buffer_info samples_buffer = samples.request();
  float* samples_ptr = static_cast< float* >(samples_buffer.ptr);   
  ssize_t s_rows = samples.shape(0);
  ssize_t s_cols = samples.shape(1);
  float** samples_out = new float*[s_rows];
  for (ssize_t i = 0; i < s_rows; ++i) {
    samples_out[i] = samples_ptr + i * s_cols;
  }

  // TODO: add parameters, allow arbitrary B
  auto B = Eigen::SparseMatrix< float >(matrix->rows(), matrix->cols());
  B.setIdentity();
  auto lo = SparseEigenAffineOperator(*matrix, B, 0.0);
  
  // Release the GIL before parallel computation
  {
    py::gil_scoped_release gil_release;
    trace_estimator< false, float >(
      &lo, params, num_inqueries, matrix_function, 
      exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, 
      error_atol, error_rtol, confidence, outlier, 
      num_threads_, 
      trace_out, error_out, samples_out, processed_samples_indices_out, num_samples_used_out, num_outliers_out, converged_out,
      alg_wall_time
    );
  }
  
  // delete[] samples_out;
  return alg_wall_time; 
}

void parallel_computation(py::array_t<double>& result_array, int num_threads) {
  int num_threads_ = num_threads;
  if (num_threads_ < 1) {
    num_threads_ = omp_get_max_threads();
  }
  num_threads_ = std::max(num_threads_, 1);
  omp_set_num_threads(num_threads);

  // Get a pointer to the underlying data of the NumPy array
  auto result_ptr = static_cast< double* >(result_array.request().ptr);

  // Get the size of the result array
  size_t size = result_array.size();

  // Release the GIL before parallel computation
  {
      py::gil_scoped_release gil_release;

      // Parallel computation using OpenMP
      #pragma omp parallel for
      for (size_t i = 0; i < size; ++i) {
          // Compute some value (e.g., i * 2.0) and store it in the result array
          result_ptr[i] = i * 2.0;
      }
  }
}

PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";
  m.def("trace_estimator_slq", &trace_estimator_slq_py2);
  m.def("trace_eigen_identity", &trace_eigen_identity);
  m.def("trace_eigen_sqrt", &trace_eigen_sqrt);
  m.def("trace_eigen_smoothstep", &trace_eigen_smoothstep);
  m.def("parallel_computation", &parallel_computation, "Perform parallel computation using OpenMP");
};