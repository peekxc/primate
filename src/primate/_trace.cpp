#include "_lanczos/lanczos.h"
#include "_trace/girard.h"
#include "eigen_operators.h"  // eigen_< mat >_wrappers
#include "spectral_functions.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

// Template function for generating module definitions for a given Operator / precision 
template< std::floating_point F, class Matrix, typename WrapperFunc >
requires std::invocable< WrapperFunc, const Matrix* >
void _trace_wrapper(py::module& m, WrapperFunc wrap = std::identity()){
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;

  m.def("hutch", [wrap](
    const Matrix* A, 
    const int nv, const int dist, const int engine_id, const int seed,
    const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
    const F atol, const F rtol, 
    const int num_threads, 
    const bool use_clt, 
    const py::kwargs& kwargs
  ) -> py_array< F > {
    if (!kwargs.contains("function")){
      throw std::invalid_argument("No matrix function supplied");
    }
    const auto op = wrap(A);
    auto rbg = ThreadedRNG64(num_threads, engine_id, seed);
    auto matrix_func = kwargs_map["function"].cast< std::string >(); 
    auto estimates = static_cast< ArrayF >(ArrayF::Zero(nv));
    
    if (matrix_func == "identity"){
      girard(A, rng, nv, dist, engine_id, seed, atol, rtol, num_threads, use_clt, estimates.data());
    } else {
      if (ncv < 2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2."); }
      if (ncv < orth+2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2+orth."); }
      const auto sf = param_spectral_func< F >(kwargs);
      const auto M = MatrixFunction(A, sf, lanczos_degree, lanczos_rtol, orth, ncv);
      girard(M, rng, nv, dist, engine_id, seed, atol, rtol, num_threads, use_clt, estimates.data());
    }
  });
}

// const Matrix& A, std::function< F(F) > fun, int lanczos_degree, F lanczos_rtol, int _orth, int _ncv

// Turns out using py::call_guard<py::gil_scoped_release>() just causes everthing to crash immediately
PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";
  _trace_wrapper(m, eigen_dense_wrapper< float >)

  // // LinearOperator exports
  // _trace_wrapper< false, float, py::object >(m, linearoperator_wrapper< float >);
  // _trace_wrapper< false, double, py::object >(m, linearoperator_wrapper< double >);
  ;
};