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
namespace py = pybind11;

using py_arr_f = py::array_t< float, py::array::c_style | py::array::forcecast >;

void eigh_tridiagonal_py(const py_arr_f& diagonals, const py_arr_f& subdiagonals, py_arr_f& eigenvectors){
  auto* diagonals_ = static_cast< float *>(diagonals.request().ptr);
  auto* subdiagonals_ = static_cast< float* >(subdiagonals.request().ptr);
  auto* out_ev = static_cast< float *>(eigenvectors.request().ptr);
  auto dim_0 = static_cast< LongIndexType >(eigenvectors.shape(0));
  eigh_tridiagonal< float >(diagonals_, subdiagonals_, out_ev, dim_0);
}

void svd_bidiagonal_py(const py_arr_f& diagonals, const py_arr_f& subdiagonals, py_arr_f& left_sv, py_arr_f& right_sv){
  auto* diagonals_ = static_cast< float *>(diagonals.request().ptr);
  auto* subdiagonals_ = static_cast< float* >(subdiagonals.request().ptr);
  auto* out_lv = static_cast< float *>(left_sv.request().ptr);
  auto* out_rv = static_cast< float *>(right_sv.request().ptr);
  auto dim_0 = static_cast< LongIndexType >(left_sv.shape(0));
  svd_bidiagonal< float >(diagonals_, subdiagonals_, out_lv, out_rv, dim_0);
}

IndexType golub_kahan_bidiagonalize_py(PyAdjointOperator< float >* op, const py_arr_f& v, const float lanczos_tol, const int orthogonalize, py_arr_f& alpha, py_arr_f& beta){
  std::pair< size_t, size_t > shape = op->shape();
  auto n = static_cast< long int >(shape.first);
  auto m = static_cast< int >(shape.second);
  auto alpha_out = static_cast< float* >(alpha.request().ptr);
  auto beta_out = static_cast< float* >(beta.request().ptr);
  golub_kahn_bidiagonalization< PyAdjointOperator< float >, float >(op, v.data(), n, m, lanczos_tol, orthogonalize, alpha_out, beta_out);
  return 0; 
}

struct DiagonalOperator {
  using value_type = float; 
  vector< float > _diagonals; 
  DiagonalOperator(vector< float > d) : _diagonals(d.begin(), d.end()) {}
  auto matvec(const float* inp, float* out) const -> void {
    std::transform(inp, inp + _diagonals.size(), _diagonals.begin(), out, std::multiplies< float >());
  }
  auto shape() const -> pair< size_t, size_t > { 
    return std::make_pair(_diagonals.size(), _diagonals.size());
  }
};

void test_lanczos(py::array_t< float > d, py::array_t< float > v, const float tol, const int orth, py_arr_f& alpha, py_arr_f& beta){
  // auto buffer = d.request();
  auto n = static_cast< long int >(d.shape(0)); 
  // auto m = static_cast< int >(d.shape(1));
  const size_t n_elems = static_cast< size_t >(n);
  std::vector< float > diag_(d.data(), d.data() + n_elems);
  auto op = DiagonalOperator(diag_);
  // auto alpha_out = std::vector< float >(n, 0.0);
  // auto beta_out = std::vector< float >(n, 0.0);
  auto alpha_out = static_cast< float* >(alpha.request().ptr);
  auto beta_out = static_cast< float* >(beta.request().ptr);
  lanczos_tridiagonalization< DiagonalOperator, float >(&op, v.data(), n, n, tol, orth, alpha_out, beta_out);
}

// Wraps an arbitrary Python object as a linear operator.
// The object must have a matvec and shape attribute. 
IndexType lanczos_tridiagonalize_py(const py::object& op, const py_arr_f& v, const float lanczos_tol, const int orthogonalize, py_arr_f& alpha, py_arr_f& beta){
  auto lo = PyLinearOperator< float >(op);
  std::pair< size_t, size_t > shape = lo.shape();
  auto n = static_cast< long int >(shape.first);
  auto m = static_cast< int >(shape.second);
  if (n != m){ throw std::invalid_argument("The Lanczos iterations only works with square operators!"); }
  auto alpha_out = static_cast< float* >(alpha.request().ptr);
  auto beta_out = static_cast< float* >(beta.request().ptr);
  if (alpha.shape(0) < m || beta.shape(0) < m){
    throw std::invalid_argument("Ouputs arrays 'alpha' / 'beta' must ");
  }
  lanczos_tridiagonalization< PyLinearOperator< float >, float >(&lo, v.data(), n, m, lanczos_tol, orthogonalize, alpha_out, beta_out);
  return 0; 
}


// See virtual overriding: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
PYBIND11_MODULE(_diagonalize, m) {
  m.def("eigh_tridiagonal", &eigh_tridiagonal_py);
  m.def("svd_bidiagonal", &svd_bidiagonal_py);
  m.def("lanczos_tridiagonalize", &lanczos_tridiagonalize_py);
  m.def("golub_kahan_bidiagonalize", &golub_kahan_bidiagonalize_py);
  m.def("test_lanczos", &test_lanczos);
}
