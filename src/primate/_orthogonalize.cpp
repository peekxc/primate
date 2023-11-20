#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "_orthogonalize/orthogonalize.h"

namespace py = pybind11;

// Note we enforce fortran style ordering here
template< std::floating_point F > 
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

template< std::floating_point F >
void _orthogonalize(py::module &m){
  m.def("mgs", &modified_gram_schmidt< F >);
  m.def("orth_vector", &orth_vector< F >);
}

PYBIND11_MODULE(_orthogonalize, m) {
  m.doc() = "orthogonalization module"; 
  _orthogonalize< float >(m);
  _orthogonalize< double >(m);
}