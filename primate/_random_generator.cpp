#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random> // mt

#include <_random_generator/vector_generator.h>
#include <_random_generator/threadedrng64.h>  // ThreadedRNG64
#include <_random_generator/pcg_random.h>     // pcg64 
#include <_random_generator/split_mix64.h>    // SplitMix64
#include <_random_generator/xoshiro_256_star_star.h>    // SplitMix64

namespace py = pybind11;

template< std::floating_point F > 
using py_array = py::array_t< F, py::array::c_style >;

// Instantiates the function templates for generic generators
template< LightRandom64Engine RNE, std::floating_point F >
void _random_generator(py::module& m, std::string suffix){
  m.def((std::string("rademacher") + suffix).c_str(), [](py_array< F >& out, const IndexType num_threads = 1){
    auto rbg = ThreadedRNG64< RNE >(num_threads);
    auto* data = static_cast< F *>(out.request().ptr);
    auto array_sz = static_cast< LongIndexType >(out.size());
    generate_array< 0, F >(rbg, data, array_sz, num_threads); 
  });
  m.def((std::string("normal") + suffix).c_str(), [](py_array< F >& out, const IndexType num_threads = 1){
    auto rbg = ThreadedRNG64< RNE >(num_threads);
    auto* data = static_cast< F *>(out.request().ptr);
    auto array_sz = static_cast< LongIndexType >(out.size());
    generate_array< 1, F >(rbg, data, array_sz, num_threads); 
  });
  // TODO: revisit this one
  // m.def((std::string("rayleigh") + suffix).c_str(), [](py_array< F >& out, const IndexType num_threads = 1){
  //   auto rbg = ThreadedRNG64< RNE >(num_threads);
  //   auto* data = static_cast< F *>(out.request().ptr);
  //   auto array_sz = static_cast< LongIndexType >(out.size());
  //   generate_array< 2, F >(rbg, data, array_sz, num_threads); 
  // });
}

// This is technically a flawed/biased generator, but with 64-bits of entropy it may be ok for many applications
// and it should be blazing fast besides
// From: https://www.pcg-random.org/posts/cpp-seeding-surprises.html
using knuth_lcg = std::linear_congruential_engine< uint64_t, 6364136223846793005U, 1442695040888963407U, 0U>;

PYBIND11_MODULE(_random_generator, m) {
  _random_generator< SplitMix64, float >(m, std::string("_sx")); 
  _random_generator< Xoshiro256StarStar, float >(m, std::string("_xs")); 
  _random_generator< std::mt19937_64, float >(m, std::string("_mt")); 
  _random_generator< pcg64, float >(m, std::string("_pcg")); 
  _random_generator< knuth_lcg, float >(m, std::string("_lcg")); 
}