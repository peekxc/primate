#include <pybind11/pybind11.h>

#include "_lanczos/lanczos.h"
#include "_trace/hutch.h"
#include "_random_generator/threadedrng64.h"
#include "eigen_operators.h"    // Eigen wrappers
#include "pylinop.h"            // py::object LinearOperator wrapper
#include "spectral_functions.h" // to parameterize the functions

namespace py = pybind11;
using namespace pybind11::literals; 

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;


// Template function for generating module definitions for a given Operator / precision 
template< bool multithreaded, std::floating_point F, class Matrix, LinearOperator Wrapper >
void _trace_wrapper(py::module& m){
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  using VectorF = Eigen::Matrix< F, Dynamic, 1 >;
  
  m.def("hutch", [](
    const Matrix& A, 
    const int nv, const int dist, const int engine_id, const int seed,
    const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv, const int method, 
    const F atol, const F rtol, 
    const int num_threads, 
    const bool use_clt, 
    const py_array< F >& t_scores, 
    const F z, 
    const py::kwargs& kwargs
  ) -> py::dict {
    if (!kwargs.contains("function")){
      throw std::invalid_argument("No matrix function supplied");
    }
    const auto op = Wrapper(A);
    const auto matrix_func = kwargs["function"].template cast< std::string >(); 
    const auto num_threads_ = get_num_threads(multithreaded ? num_threads : 1); 
    // std::cout << "Number of threads: " << num_threads_ << std::endl;

    auto rng = ThreadedRNG64(num_threads_, engine_id, seed);
    auto estimates = static_cast< ArrayF >(ArrayF::Zero(nv));
    auto mu_est = F(0.0);
    auto wall_time = size_t(0);

    // t.ppf(0.975, df=np.arange(30)+1)
    // const auto z = std::sqrt(2.0) * erf_inv< 3 >(double(0.95));
    if (matrix_func == "None"){
      mu_est = hutch< F >(op, rng, nv, dist, engine_id, seed, atol, rtol, num_threads_, use_clt, t_scores.data(), z, estimates.data(), wall_time);
    } else {
      if (ncv < 2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2."); }
      if (ncv < orth+2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2+orth."); }
      bool is_native = true; 
      const auto sf = param_vector_func< F >(kwargs, is_native);
      const auto M = MatrixFunction(op, sf, lanczos_degree, lanczos_rtol, orth, ncv, is_native, static_cast< weight_method >(method));
      mu_est = hutch< F >(M, rng, nv, dist, engine_id, seed, atol, rtol, num_threads_, use_clt, t_scores.data(), z, estimates.data(), wall_time);
    }
    return py::dict(
      "estimate"_a=mu_est, 
      "samples"_a=estimates, 
      "total_time_us"_a = wall_time, 
      "matvec_time_us"_a = op.matvec_time / num_threads_
    );
  });

  // Computes the trace of Q.T @ (A @ Q) including the inner terms q_i^T A q_i 
  m.def("quad_sum", [](const Matrix& A, DenseMatrix< F > Q) -> py::tuple {
    const auto op = Wrapper(A);
    F quad_sum = 0.0; 
    const size_t N = static_cast< size_t >(Q.cols());
    auto estimates = static_cast< ArrayF >(ArrayF::Zero(N));
    auto y = static_cast< VectorF >(VectorF::Zero(Q.rows()));
    
    for (size_t j = 0; j < N; ++j){
      op.matvec(Q.col(j).data(), y.data());
      estimates[j] = Q.col(j).adjoint().dot(y);
      quad_sum += estimates[j];
    }
    return py::make_tuple(quad_sum, py::cast(estimates));
  });
}

// const Matrix& A, std::function< F(F) > fun, int lanczos_degree, F lanczos_rtol, int _orth, int _ncv

// Turns out using py::call_guard<py::gil_scoped_release>() just causes everthing to crash immediately
PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";
  _trace_wrapper< true, float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m);
  _trace_wrapper< true, double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m);
  
  _trace_wrapper< true, float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float > >(m);
  _trace_wrapper< true, double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double > >(m);
  
  // Note we cannot multi-thread arbitrary calls to Python due to the GIL
  _trace_wrapper< false, float, py::object, PyLinearOperator< float > >(m);
  _trace_wrapper< false, double, py::object, PyLinearOperator< double > >(m);

  // Wrapping MatrixFunctions natively extends all trace functionality
  // _trace_wrapper< true, float, DenseMatrix< float >, MatrixFunction< float, DenseEigenLinearOperator< float > > >(m);
  // _trace_wrapper< false, double, py::object, PyLinearOperator< double > >(m);

  // // LinearOperator exports
  // _trace_wrapper< false, float, py::object >(m, linearoperator_wrapper< float >);
  // _trace_wrapper< false, double, py::object >(m, linearoperator_wrapper< double >);
  ;
};