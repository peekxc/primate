#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "_orthogonalize/orthogonalize.h"

namespace py = pybind11;

// Note we enforce fortran style ordering here
template< std::floating_point F > 
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

// template< std::floating_point F >
// auto orth_poly(F x, int i, F mu_sqrt, const F* a, const F* b, const int n) noexcept -> F {
//   if (i < 0){ 
//     return 0.0; 
//   } else if (i == 0){
//     return 1 / mu_sqrt;
//   } else if (i == 1){
//     return (x - a[0]) * (1 / mu_sqrt) / b[1];
//   } else if (i < n){
//     F z = (x - a[i-1]) * orth_poly(x, i - 1, mu_sqrt, a, b, n);
//     z -= b[i-1] * orth_poly(x, i - 2, mu_sqrt, a, b, n);
//     z /= b[i];
//     return z; 
//   } else {
//     return 0; 
//   }
// }


template< std::floating_point F >
void _orthogonalize(py::module &m){
  m.def("mgs", &modified_gram_schmidt< F >);
  m.def("orth_vector", &orth_vector< F >);
  // m.def("orth_poly", &orth_poly< F >);
}

PYBIND11_MODULE(_orthogonalize, m) {
  m.doc() = "orthogonalization module"; 
  _orthogonalize< float >(m);
  _orthogonalize< double >(m);
}