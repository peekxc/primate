#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include "../_definitions/types.h"
#include <_random_generator/random_number_generator.h>
#include <_random_generator/random_array_generator.h>
namespace py = pybind11;


void rademacher(py::array_t< float, py::array::c_style> out, const IndexType num_threads = 1){
  auto rng = RandomNumberGenerator();
  auto* data = static_cast< float *>(out.request().ptr);
  auto array_sz = static_cast< LongIndexType >(out.size());
  RandomArrayGenerator< float >::generate_random_array(rng, data, array_sz, num_threads); 
}


PYBIND11_MODULE(_random_generator, m) {
  m.def("rademacher", &rademacher);
  // m.def("rademacher", &rademacher, py::call_guard<py::gil_scoped_release>());
}