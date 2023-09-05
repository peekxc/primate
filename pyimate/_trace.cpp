#include <pybind11/pybind11.h>

#include "eigen_operators.h"
#include <_definitions/types.h>
#include <_trace_estimator/trace_estimator.h>
// #include <_functions/functions.h>
#include <_diagonalization/lanczos_tridiagonalization.h>


namespace py = pybind11;
using py_arr_f = py::array_t< float >;
// class SmoothStepEps : public Function {
//   public:
//     SmoothStepEps(){
//       eps = 0; 
//     }
//     virtual float function(const float lambda_) const {
//       return 0;
//     };
//     virtual double function(const double lambda_) const {
//       return 0;
//     }
//     virtual long double function(const long double lambda_) const {
//       return 0;
//     }
//     double eps;
// };


// int apply_smoothstep(int i, int j) {
//   auto S = SmoothStepEps();
//   return i + j;
// }

float trace_estimator_slq_py(
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
  trace_estimator< false, float >(
    &lo, params, num_inqueries, matrix_function, 
    exponent, orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, 
    error_atol, error_rtol, confidence, outlier, 
    num_threads_, 
    trace_out, error_out, samples_out, processed_samples_indices_out, num_samples_used_out, num_outliers_out, converged_out,
    alg_wall_time
  );
  
  delete[] samples_out;
  return alg_wall_time; 
}

PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";
  m.def("trace_estimator_slq", &trace_estimator_slq_py);
};