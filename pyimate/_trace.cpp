#include <pybind11/pybind11.h>
#include <functions/functions.h>
// #include <_c_trace_estimator/c_trace_estimator.h>

// #include <_c_linear_operator/c_linear_operator.h>
// #include "_random_generator/"



class SmoothStepEps : public Function {
  public:
    SmoothStepEps(){
      eps = 0; 
    }
    virtual float function(const float lambda_) const {
      return 0;
    };
    virtual double function(const double lambda_) const {
      return 0;
    }
    virtual long double function(const long double lambda_) const {
      return 0;
    }
    double eps;
};

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


PYBIND11_MODULE(_trace, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("apply_smoothstep", &apply_smoothstep);
}