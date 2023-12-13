#ifndef _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_

#include <random>   // uniform_random_bit_generator, normal_distribution
#include <cstdint>  // uint64_t
#include <cmath> // isnan
#include <bitset>   // bitset
#include "omp_support.h" // conditionally enables openmp pragmas
#include "threadedrng64.h"	// ThreadSafeRBG
#include "rne_engines.h"		// all the engines

using IndexType = unsigned int;
using LongIndexType = unsigned long;
static constexpr long num_bits = 64; 

enum Distribution { rademacher = 0, normal = 1, rayleigh = 2 };

// SIMD-vectorized rademacher vector generation; makes N // 64 calls to random bit generator
template< std::floating_point F, ThreadSafeRBG RBG > 
void generate_rademacher(const long n, RBG& random_bit_generator, const int thread_id, F* array, F& arr_norm){
	const auto N = static_cast< size_t >(n / RBG::num_bits);
	for (size_t i = 0; i < N; ++i) {
		std::bitset< RBG::num_bits > ubits { random_bit_generator.next(thread_id) };
		#pragma omp simd 
		for (size_t j = 0; j < RBG::num_bits; ++j) {
			array[i*RBG::num_bits + j] = 2 * int(ubits[j]) - 1;
		}
	}
	// This loop should have less than 64 iterations.
	std::bitset< RBG::num_bits > ubits { random_bit_generator.next(thread_id) };
	for (size_t j = N * RBG::num_bits, i = 0; j < size_t(n); ++j, ++i){
		array[j] = (2 * int(ubits[i]) - 1);
	}
	arr_norm = static_cast< F >(std::sqrt(n)); 
}

// Zero mean, unit variance gaussian
template< std::floating_point F, ThreadSafeRBG RBG > 
void generate_normal(const unsigned long n, RBG& bit_generator, const int thread_id, F* array, F& arr_norm){
	static std::normal_distribution d { 0.0, 1.0 };
	// const auto N = static_cast< unsigned long >(n / RBG::num_bits);
	auto& gen = *bit_generator.generators[thread_id];
	// #pragma omp simd
	for (auto i = static_cast< unsigned long >(0); i < n; ++i){
		array[i] = d(gen);
	}
	// // This loop should have less than num_bits iterations.
	// for (auto j = static_cast< unsigned long >(N * RBG::num_bits), i = static_cast< unsigned long >(0); j < n; ++j, ++i){
	// 	array[j] = d(gen);
	// }
	
	arr_norm = 0.0; 
	// #pragma omp simd reduction(+:arr_norm)
	for (unsigned long i = 0; i < n; ++i){
		arr_norm += std::abs(array[i]);
	}
	arr_norm = std::sqrt(arr_norm);
}

// Generates an isotropic random vector for a given thread id
template< std::floating_point F, ThreadSafeRBG RBG > 
void generate_isotropic(const Distribution dist_id, const long n, RBG& rbg, const int thread_id, F* array, F& arr_norm){
	switch(dist_id){
		case rademacher:
			generate_rademacher(n, rbg, thread_id, array, arr_norm);
			break; 
		case normal: 
			generate_normal(n, rbg, thread_id, array, arr_norm);
			break; 
		case rayleigh: 
			// generate_rayleigh(n, bit_generator, thread_id, array, arr_norm);
			break; 
	}

	// Ensure the array has unit norm
	const F arr_norm_inv = 1.0 / arr_norm;
	#pragma omp simd
	for (auto i = static_cast< long >(0); i < n; ++i){
		array[i] *= arr_norm_inv;
	}
}
#endif 