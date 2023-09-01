#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseCore> // SparseMatrix

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

// void trace_estimator(){
//   auto S = SmoothStepEps();
//   auto est = cTraceEstimator< float >();
//   est.c_trace_estimator(
//     cLinearOperator<DataType>* A,
//     DataType* parameters,
//     const IndexType num_inquiries,
//     const Function* matrix_function,
//     const FlagType gram,
//     const DataType exponent,
//     const FlagType orthogonalize,
//     const IndexType lanczos_degree,
//     const DataType lanczos_tol,
//     const IndexType min_num_samples,
//     const IndexType max_num_samples,
//     const DataType error_atol,
//     const DataType error_rtol,
//     const DataType confidence_level,
//     const DataType outlier_significance_level,
//     const IndexType num_threads,
//     DataType* trace,
//     DataType* error,
//     DataType** samples,
//     IndexType* processed_samples_indices,
//     IndexType* num_samples_used,
//     IndexType* num_outliers,
//     FlagType* converged,
//     float& alg_wall_time
//   );

// }


int apply_smoothstep(int i, int j) {
  auto S = SmoothStepEps();
  return i + j;
}

void trace_estimator_py(
  const Eigen::SparseMatrix< float >* matrix, 
  const py_arr_f& parameters, 
  const size_t num_inqueries
  const bool gramian, 
  const float exponent, 
  const int orthogonalize, 
  const size_t lanczos_degree, 
  const size_t min_num_samples, 
  const size_t max_num_samples, 
  const float error_atol, 
  const float error_rtol,
  const float confidence, 
  const size_t num_threads, 
){
  float* params = static_cast< float* >(parameters.request().ptr);
  auto matrix_function = [](float eigenvalue) -> float { return eigenvalue; }; 
  DataType* trace,
  DataType* error,
  DataType** samples,
  IndexType* processed_samples_indices,
  IndexType* num_samples_used,
  IndexType* num_outliers,
  FlagType* converged,
  float alg_wall_time = 0.0; 
  trace_estimator(
    matrix, params, num_inqueries, matrix_function, 
    (const FlagType) gramian, exponent, orthogonalize, lanczos_degree, min_num_samples, max_num_samples, 
    error_atol, error_rtol, confidence, 1.0 - confidence, 
    num_threads, 

    &alg_wall_time
  )
}

PYBIND11_MODULE(_trace, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("apply_smoothstep", &apply_smoothstep);
}