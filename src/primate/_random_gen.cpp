#include <_random_generator/random_concepts.h>
#include <_random_generator/rne_engines.h>
#include <_random_generator/threadedrng64.h>
#include <_random_generator/vector_generator.h>

// #define USE_NANOBIND 1
#undef USE_NANOBIND
#ifdef USE_NANOBIND
  #include <nanobind/nanobind.h>
  #include <nanobind/ndarray.h>
#else
  #include <pybind11/pybind11.h>
  #include <pybind11/numpy.h>
#endif

#ifdef USE_NANOBIND
  namespace nb = nanobind;
  template< std::floating_point F > 
  using py_array = nb::ndarray< F >;
  using py_module = nb::module_;
#else 
  namespace py = pybind11;
  template< std::floating_point F > 
  using py_array = py::array_t< F, py::array::c_style >;
  using py_module = py::module;
#endif

// Instantiates the function templates for generic generators
template< std::floating_point F >
void _random(py_module& m){
  m.def("rademacher", [](py_array< F >& out, const int rng = 0, const int seed = -1){
    auto rbg = ThreadedRNG64(1, rng, seed);
    auto* data = out.mutable_data();
    auto array_sz = static_cast< size_t >(out.size());
    F arr_norm = 0.0; 
    generate_rademacher(array_sz, rbg, 0, data, arr_norm);
  });
  m.def("normal", [](py_array< F >& out, const int rng = 0, const int seed = -1){
    auto rbg = ThreadedRNG64(1, rng, seed);
    auto* data = out.mutable_data();
    auto array_sz = static_cast< size_t >(out.size());
    F arr_norm = 0.0; 
    generate_normal(array_sz, rbg, 0, data, arr_norm);
  });
}

// #ifdef USE_NANOBIND
// NB_MODULE(_random_gen, m)
// #else 
// PYBIND11_MODULE(_random_gen, m)
// #endif
PYBIND11_MODULE(_random_gen, m){
  _random< float >(m); 
  _random< double >(m); 
}