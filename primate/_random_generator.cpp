#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include "../_definitions/types.h"
#include <random> // mt
// #include <_random_generator/random_number_generator.h>

#include <_random_generator/vector_generator.h>
#include <_random_generator/threadedrng64.h>  // ThreadedRNG64
#include <_random_generator/pcg_random.h>     // pcg64 
#include <_random_generator/split_mix64.h>    // SplitMix64
#include <_random_generator/xoshiro_256_star_star.h>    // SplitMix64

namespace py = pybind11;


// void rademacher_xoshiro256(py::array_t< float, py::array::c_style> out, const IndexType num_threads = 1){
//   auto rng = RandomNumberGenerator();
//   auto* data = static_cast< float *>(out.request().ptr);
//   auto array_sz = static_cast< LongIndexType >(out.size());
//   RandomArrayGenerator< float >::generate_random_array(rng, data, array_sz, num_threads); 
// }


// void rademacher_pcg(py::array_t< float, py::array::c_style> out, const IndexType num_threads = 1){
//   // pcg_extras::seed_seq_from< std::random_device > seed_source;
//   // pcg64 rng(seed_source);
//   auto rbg = ThreadedRNG64< pcg64 >(num_threads); 
//   auto* data = static_cast< float *>(out.request().ptr);
//   auto array_sz = static_cast< LongIndexType >(out.size());
//   VectorGenerator< float >::generate_array(rbg, data, array_sz, num_threads); 
// }

// By not by reference?
void rademacher_mt(py::array_t< float, py::array::c_style >& out, const IndexType num_threads = 1){
  // pcg_extras::seed_seq_from< std::random_device > seed_source;
  // std::mt19937_64 rng(seed_source);
  auto rbg = ThreadedRNG64< std::mt19937_64 >(num_threads);
  auto* data = static_cast< float *>(out.request().ptr);
  auto array_sz = static_cast< LongIndexType >(out.size());
  VectorGenerator< float >::generate_array(rbg, data, array_sz, num_threads); 
}

void rademacher_sx(py::array_t< float, py::array::c_style >& out, const IndexType num_threads = 1){
  auto rbg = ThreadedRNG64< SplitMix64 >(num_threads);
  auto* data = static_cast< float *>(out.request().ptr);
  auto array_sz = static_cast< LongIndexType >(out.size());
  VectorGenerator< float >::generate_array(rbg, data, array_sz, num_threads); 
}

void rademacher_xs(py::array_t< float, py::array::c_style >& out, const IndexType num_threads = 1){
  auto rbg = ThreadedRNG64< Xoshiro256StarStar >(num_threads);
  auto* data = static_cast< float *>(out.request().ptr);
  auto array_sz = static_cast< LongIndexType >(out.size());
  VectorGenerator< float >::generate_array(rbg, data, array_sz, num_threads); 
}


// void rademacher_pcg_single(py::array_t< float, py::array::c_style> out){
//   pcg_extras::seed_seq_from< std::random_device > seed_source;
//   pcg32 rng(seed_source);
//   const auto a = std::numeric_limits< uint64_t >::min(); 
//   const auto b = std::numeric_limits< uint64_t >::max(); 
//   std::uniform_int_distribution< uint64_t > dist(a, b);
//   auto* data = static_cast< float *>(out.request().ptr);

//   // Unrolled version that uses uniform int
//   const auto array_size = static_cast< size_t >(out.size());
//   const size_t num_bits = 64;
//   const size_t inc = size_t(array_size / num_bits);
//   for (auto i = 0; i < size_t(array_size / num_bits); ++i) {
//     std::bitset< 64 > ubits { dist(rng) };
//     for (auto j = 0; j < num_bits; ++j) {
//       data[i*num_bits + j] = ubits[j] ? 1.0 : -1.0; 
//     }
//   }
//   // This loop should have less than 64 iterations.
//   std::bitset< 64 > ubits{ dist(rng) };
//   for (auto j = inc * num_bits, i = size_t(0); j < array_size; ++j, ++i){
//     data[j] = ubits[i] ? 1.0 : -1.0; 
//   }
// }


PYBIND11_MODULE(_random_generator, m) {
  // m.def("rademacher_xoshiro256", &rademacher_xoshiro256);
  // m.def("rademacher_pcg", &rademacher_pcg);
  m.def("rademacher_mt", &rademacher_mt); // py::call_guard<py::gil_scoped_release>()
  m.def("rademacher_sx", &rademacher_sx); // py::call_guard<py::gil_scoped_release>()
  m.def("rademacher_xs", &rademacher_xs); //py::call_guard<py::gil_scoped_release>() 
  // m.def("rademacher", &rademacher, py::call_guard<py::gil_scoped_release>());
}