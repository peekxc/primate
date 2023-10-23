#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include "../_definitions/types.h"
#include <_definitions/definitions.h>
#include <_definitions/types.h>
#include <_diagonalization/diagonalization.h>
#include <_diagonalization/lanczos_tridiagonalization.h>
#include <_diagonalization/golub_kahn_bidiagonalization.h>

#include "pylinops.h"
#include "eigen_operators.h"
namespace py = pybind11;

template< std::floating_point F >
using py_array = py::array_t< F, py::array::c_style | py::array::forcecast >;

template< std::floating_point F >
void eigh_tridiagonal_py(const py_array< F >& diagonals, const py_array< F >& subdiagonals, py_array< F >& eigenvectors){
  auto* diagonals_ = static_cast< F *>(diagonals.request().ptr);
  auto* subdiagonals_ = static_cast< F* >(subdiagonals.request().ptr);
  auto* out_ev = static_cast< F *>(eigenvectors.request().ptr);
  auto dim_0 = static_cast< LongIndexType >(eigenvectors.shape(0));
  eigh_tridiagonal< F >(diagonals_, subdiagonals_, out_ev, dim_0);
}

template< std::floating_point F >
void svd_bidiagonal_py(const py_array< F >& diagonals, const py_array< F >& subdiagonals, py_array< F >& left_sv, py_array< F >& right_sv){
  auto* diagonals_ = static_cast< F *>(diagonals.request().ptr);
  auto* subdiagonals_ = static_cast< F* >(subdiagonals.request().ptr);
  auto* out_lv = static_cast< F *>(left_sv.request().ptr);
  auto* out_rv = static_cast< F *>(right_sv.request().ptr);
  auto dim_0 = static_cast< LongIndexType >(left_sv.shape(0));
  svd_bidiagonal< F >(diagonals_, subdiagonals_, out_lv, out_rv, dim_0);
}

template< typename Matrix, std::floating_point F = typename Matrix::value_type >
IndexType lanczos_tridiagonalize_api(Matrix* op, const py_array< F >& v, const int num_steps, const float lanczos_tol, const int orthogonalize, py_array< F >& alpha, py_array< F >& beta){
  std::pair< size_t, size_t > shape = op->shape();
  auto n = static_cast< long int >(shape.first);
  auto m = static_cast< int >(shape.second);
  if (n != m){ throw std::invalid_argument("The Lanczos iterations only works with square operators!"); }
  auto alpha_out = static_cast< F* >(alpha.request().ptr);
  auto beta_out = static_cast< F* >(beta.request().ptr);
  if (alpha.shape(0) < num_steps || beta.shape(0) < num_steps){
    throw std::invalid_argument("Output arrays 'alpha' / 'beta' must at least match the number of steps.");
  }
  lanczos_tridiagonalization< F, Matrix >(op, v.data(), n, num_steps, lanczos_tol, orthogonalize, alpha_out, beta_out);
  return 0; 
}

template< std::floating_point F >
IndexType golub_kahan_bidiagonalize_py(PyAdjointOperator< F >* op, const py_array< F >& v, const F lanczos_tol, const int orthogonalize, py_array< F >& alpha, py_array< F >& beta){
  std::pair< size_t, size_t > shape = op->shape();
  auto n = static_cast< long int >(shape.first);
  auto m = static_cast< int >(shape.second);
  auto alpha_out = static_cast< F* >(alpha.request().ptr);
  auto beta_out = static_cast< F* >(beta.request().ptr);
  golub_kahn_bidiagonalization< F, PyAdjointOperator< F > >(op, v.data(), n, m, lanczos_tol, orthogonalize, alpha_out, beta_out);
  return 0; 
}


template< std::floating_point F >
void _diagonalize(py::module &m){
  m.def("eigh_tridiagonal", &eigh_tridiagonal_py< F >);
  m.def("svd_bidiagonal", &svd_bidiagonal_py< F >);
  m.def("golub_kahan_bidiagonalize", &golub_kahan_bidiagonalize_py< F >);
  
  // Lanczos overloads
  m.def("lanczos_tridiagonalize", [](const Eigen::SparseMatrix< F >& mat, const py_array< F >& v, const int num_steps, const F lanczos_tol, const int orthogonalize, py_array< F >& alpha, py_array< F >& beta){
    auto lo = SparseEigenLinearOperator(mat);
    lanczos_tridiagonalize_api(&lo, v, num_steps, lanczos_tol, orthogonalize, alpha, beta);
  });
  m.def("lanczos_tridiagonalize", &lanczos_tridiagonalize_api< PyLinearOperator< F > >);
  m.def("lanczos_tridiagonalize", [](const py::object& op, const py_array< F >& v, const int num_steps, const F lanczos_tol, const int orthogonalize, py_array< F >& alpha, py_array< F >& beta){
    const auto lo = PyLinearOperator< F >(op); // attempt to wrap
    lanczos_tridiagonalize_api(&lo, v, num_steps, lanczos_tol, orthogonalize, alpha, beta);
  });
}

// Also for variable binding: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#binding-functions-with-template-parameters
// See virtual overriding: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
PYBIND11_MODULE(_diagonalize, m) {
  _diagonalize< float >(m);
  _diagonalize< double >(m);
  _diagonalize< long double >(m);
}
