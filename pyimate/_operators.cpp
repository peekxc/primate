#include "pylinops.h"

namespace py = pybind11;

IndexType lanczos_tridiagonalize_py(PyLinearOperator< float >* op, const py_arr_f& v, const float lanczos_tol, const int orthogonalize, py_arr_f& alpha, py_arr_f& beta){
  if (op == nullptr){
    throw std::invalid_argument("Linear operator is empty (nullptr). It's lifetime has not been preserved!");
  }
  std::pair< size_t, size_t > shape = op->shape();
  auto n = static_cast< long int >(shape.first);
  auto m = static_cast< int >(shape.second);
  auto alpha_out = static_cast< float* >(alpha.request().ptr);
  auto beta_out = static_cast< float* >(beta.request().ptr);
  lanczos_tridiagonalization< PyLinearOperator< float >, float >(op, v.data(), n, m, lanczos_tol, orthogonalize, alpha_out, beta_out);
  return 0; 
}

// See: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive
PYBIND11_MODULE(_operators, m) {
  // py::class_< PyDiagonalOperator< float > >(m, "PyDiagonalOperator")
  //   .def(py::init< const py::array_t< float > >())
  //   .def("matvec", (py::array_t< float >(PyDiagonalOperator< float >::*)(const py::array_t< float >&) const ) &PyDiagonalOperator< float >::matvec)
  //   .def_property_readonly("shape", &PyDiagonalOperator< float >::shape)
  //   .def_property_readonly("dtype", &PyDiagonalOperator< float >::dtype)
  //   ;
  // py::class_< PyDenseMatrix< float > >(m, "PyDenseMatrix")
  //   .def(py::init< const py::array_t< float > >())
  //   .def("matvec", (py::array_t< float >(PyDenseMatrix< float >::*)(const py::array_t< float >&) const) &PyDenseMatrix< float >::matvec)
  //   .def_property_readonly("shape", &PyDenseMatrix< float >::shape)
  //   .def_property_readonly("dtype", &PyDenseMatrix< float >::dtype)
  //   ;
  //   py::class_< PyAdjointOperator< float > >(m, "PyAdjointOperator")
  //   .def(py::init< const py::object& >(), py::keep_alive< 1, 2 >())
  //   .def("matvec", (py::array_t< float >(PyAdjointOperator< float >::*)(const py::array_t< float >&) const) &PyAdjointOperator< float >::matvec)
  //   .def("rmatvec", (py::array_t< float >(PyAdjointOperator< float >::*)(const py::array_t< float >&) const) &PyAdjointOperator< float >::rmatvec)
  //   .def_property_readonly("shape", &PyAdjointOperator< float >::shape)
  //   .def_property_readonly("dtype", &PyAdjointOperator< float >::dtype)
  //   ;
  py::class_< PyLinearOperator< float > >(m, "PyLinearOperator")
    .def(py::init< const py::object& >()) // or (2, 1)? py::keep_alive< 1, 2 >()
    .def("matvec", (py::array_t< float >(PyLinearOperator< float >::*)(const py::array_t< float >&) const) &PyLinearOperator< float >::matvec)
    .def_property_readonly("shape", &PyLinearOperator< float >::shape)
    .def_property_readonly("dtype", &PyLinearOperator< float >::dtype)
    .def("lanczos", &lanczos_tridiagonalize_py)
    ;

}

// using py::array_t< float >(PyDenseMatrix< float >::*)(const py::array_t< float >&) const;
