#include "_lanczos/lanczos.h"
#include "_trace/hutch.h"
#include "eigen_operators.h"    // Eigen wrappers
#include "pylinop.h"            // py::object LinearOperator wrapper
#include "spectral_functions.h" // to parameterize the functions

#include <pybind11/pybind11.h>
namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

// Template function for generating module definitions for a given Operator / precision 
template< bool multithreaded, std::floating_point F, class Matrix, LinearOperator Wrapper >
void _trace_wrapper(py::module& m){
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;

  m.def("hutch", [](
    const Matrix& A, 
    const int nv, const int dist, const int engine_id, const int seed,
    const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
    const F atol, const F rtol, 
    const int num_threads, 
    const bool use_clt, 
    const py::kwargs& kwargs
  ) -> py::tuple {
    if (!kwargs.contains("function")){
      throw std::invalid_argument("No matrix function supplied");
    }
    const auto op = Wrapper(A);
    const auto matrix_func = kwargs["function"].cast< std::string >(); 
    const auto num_threads_ = multithreaded ? num_threads : 1; 

    auto rng = ThreadedRNG64(num_threads_, engine_id, seed);
    auto estimates = static_cast< ArrayF >(ArrayF::Zero(nv));
    auto mu_est = F(0.0);

    if (matrix_func == "None"){
      mu_est = hutch< F >(op, rng, nv, dist, engine_id, seed, atol, rtol, num_threads_, use_clt, estimates.data());
    } else {
      if (ncv < 2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2."); }
      if (ncv < orth+2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2+orth."); }
      const auto sf = param_spectral_func< F >(kwargs);
      const auto M = MatrixFunction(op, sf, lanczos_degree, lanczos_rtol, orth, ncv);
      mu_est = hutch< F >(M, rng, nv, dist, engine_id, seed, atol, rtol, num_threads_, use_clt, estimates.data());
    }
    return py::make_tuple(mu_est, py::cast(estimates));
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
  // // LinearOperator exports
  // _trace_wrapper< false, float, py::object >(m, linearoperator_wrapper< float >);
  // _trace_wrapper< false, double, py::object >(m, linearoperator_wrapper< double >);
  ;
};