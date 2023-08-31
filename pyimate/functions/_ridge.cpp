#include <pybind11/pybind11.h>

double square(double x) {
  return x * x;
}

PYBIND11_MODULE(_ridge, m) {
  m.def("square", &square);
}