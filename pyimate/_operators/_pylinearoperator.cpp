
#include "../pylinops.h"

// #include <_definitions/definitions.h>
// #include <_definitions/types.h>
// #include <_diagonalization/diagonalization.h>
// #include <_diagonalization/lanczos_tridiagonalization.h>
// #include <_diagonalization/golub_kahn_bidiagonalization.h>
// #include <_linear_operator/linear_operator.h>
// #include <_c_basic_algebra/c_matrix_operations.h>



// See: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive
PYBIND11_MODULE(_pylinearoperator, m) {
  py::class_< PyLinearOperator< float > >(m, "PyLinearOperator")
    .def(py::init< const py::object& >()) // or (2, 1)? py::keep_alive< 1, 2 >()
    .def("matvec", (py::array_t< float >(PyLinearOperator< float >::*)(const py::array_t< float >&) const) &PyLinearOperator< float >::matvec)
    .def_property_readonly("shape", &PyLinearOperator< float >::shape)
    .def_property_readonly("dtype", &PyLinearOperator< float >::dtype)
    .def("lanczos", &lanczos_tridiagonalize_py)
    ;
}