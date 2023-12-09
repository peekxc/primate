#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "_random_generator/vector_generator.h"
#include "_lanczos/lanczos.h"
#include "eigen_operators.h"
#include <cmath> // constants
#include <iostream>
#include <stdio.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// PYBIND11_MODULE(_vapprox, m) {
//   py::class_< Pet >(m, "Pet")
//     .def(py::init<const std::string &>())
//     .def("setName", &Pet::setName)
//     .def("getName", &Pet::getName);
// }

