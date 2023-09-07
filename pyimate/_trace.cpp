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
  int num_threads_ = num_threads;
  if (num_threads_ < 1) {
    num_threads_ = omp_get_max_threads();
  }
  num_threads_ = std::max(num_threads_, 1);

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

  // TODO: add parameters, allow arbitrary B
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

template< std::floating_point DataType, Operator MatrixOp, std::invocable< DataType > Func > 
float call_trace(const MatrixOp* matrix, Func&& matrix_function, const py::args& args){
  const py_arr_f parameters = args[0].cast< py_arr_f >(); 
  const size_t num_inqueries = args[1].cast< size_t >(); 
  const float exponent = args[2].cast< float >(); 
  const int orthogonalize = args[3].cast< int >(); 
  const float lanczos_degree = args[4].cast< float >(); 
  const float lanczos_tol = args[5].cast<float>();
  const size_t min_num_samples = args[6].cast<size_t>();
  const size_t max_num_samples = args[7].cast<size_t>();
  const float error_atol = args[8].cast<float>();
  const float error_rtol = args[9].cast<float>();
  const float confidence = args[10].cast<float>();
  const float outlier = args[11].cast<float>();
  const int num_threads = args[12].cast<int>();
  py_arr_f trace = args[13].cast<py_arr_f>();
  py_arr_f error = args[14].cast<py_arr_f>();
  py_arr_f samples = args[15].cast<py_arr_f>();
  py::array_t<int> processed_samples_indices = args[16].cast<py::array_t<int>>();
  py::array_t<int> num_samples_used = args[17].cast<py::array_t<int>>();
  py::array_t<int> num_outliers = args[18].cast<py::array_t<int>>();
  py::array_t<int> converged = args[19].cast<py::array_t<int>>();

  // Make the call -- at this point the gil should be released
  float alg_time = trace_estimator_slq_py< float >(
    matrix, matrix_function, 
    parameters, num_inqueries, 
    exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, 
    error_atol, error_rtol, confidence, outlier, 
    num_threads, 
    trace, error, samples, processed_samples_indices, num_samples_used, num_outliers, converged
  );
  return alg_time; 
}

// template < typename ...Params>
// Params&&... params
// std::forward<Params>(params)...
float trace_eigen_identity(const Eigen::SparseMatrix< float >* A, const py::args& args) {
  auto B = Eigen::SparseMatrix< float >(A->rows(), A->cols());
  B.setIdentity();
  const auto lo = SparseEigenAffineOperator(*A, B, 0.0);
  const auto matrix_function = [](float eigenvalue) -> float { return eigenvalue; }; 
  float alg_time = call_trace< float >(&lo, matrix_function, args);
  return alg_time; 
}

PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";
  // m.def("trace_estimator_slq", &trace_estimator_slq_py, py::call_guard<py::gil_scoped_release>());
  m.def("trace_eigen_identity", &trace_eigen_identity, py::call_guard<py::gil_scoped_release>());  
};