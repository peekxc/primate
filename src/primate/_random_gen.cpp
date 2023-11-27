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
template< LightRandom64Engine RNE, std::floating_point F >
void _random(py_module& m, std::string suffix){
  m.def((std::string("rademacher") + suffix).c_str(), [](py_array< F >& out, const size_t num_threads = 1, const int seed = -1){
    auto rbg = ThreadedRNG64< RNE >(num_threads, seed);
    auto* data = out.mutable_data();
    auto array_sz = static_cast< size_t >(out.size());
    generate_array< 0, F >(rbg, data, array_sz, num_threads); 
  });
  m.def((std::string("normal") + suffix).c_str(), [](py_array< F >& out, const size_t num_threads = 1, const int seed = -1){
    auto rbg = ThreadedRNG64< RNE >(num_threads, seed);
    auto* data = out.mutable_data();
    auto array_sz = static_cast< size_t >(out.size());
    generate_array< 1, F >(rbg, data, array_sz, num_threads); 
  });
  // TODO: revisit this one
  // m.def((std::string("rayleigh") + suffix).c_str(), [](py_array< F >& out, const IndexType num_threads = 1){
  //   auto rbg = ThreadedRNG64< RNE >(num_threads);
  //   auto* data = static_cast< F *>(out.request().ptr);
  //   auto array_sz = static_cast< LongIndexType >(out.size());
  //   generate_array< 2, F >(rbg, data, array_sz, num_threads); 
  // });
  m.def((std::string("rademacher_simd") + suffix).c_str(), [](py_array< F >& out, const size_t num_threads = 1, const int seed = -1){
    auto rbg = ThreadedRNG64< RNE >(num_threads, seed);
    auto* data = out.mutable_data();
    auto array_sz = static_cast< size_t >(out.size());
    generate_isotropic< 0, F >(rbg, data, array_sz, 0); 
  });
}

// #ifdef USE_NANOBIND
// NB_MODULE(_random_gen, m)
// #else 
// PYBIND11_MODULE(_random_gen, m)
// #endif
PYBIND11_MODULE(_random_gen, m){
  _random< SplitMix64, float >(m, std::string("_sx")); 
  _random< Xoshiro256StarStar, float >(m, std::string("_xs")); 
  _random< std::mt19937_64, float >(m, std::string("_mt")); 
  _random< pcg64, float >(m, std::string("_pcg")); 
  _random< knuth_lcg, float >(m, std::string("_lcg")); 
}