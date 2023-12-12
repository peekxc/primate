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
// array[i*RBG::num_bits + j] = ubits[j] ? 1.0 : -1.0; // Shift checks the j-th bit (from right to left) is 1 or 0

// template< size_t Distr, std::floating_point DataType, ThreadSafeRBG RBG > 
// void generate_array(
// 	RBG& random_bit_generator,
// 	DataType* array,
// 	const LongIndexType array_size,
// 	const IndexType num_threads
// ){
// 	// Passing zero-threads indicates caller was already parallelized and thus should not re-set threads
// 	// Otherwise 'num_threads' threads is created 
// 	if (num_threads > 0) {
// 		omp_set_num_threads(num_threads);
// 	}

// 	// Get the thread id
// 	int thread_id = 0;
// 	if (num_threads == 0) { // if in parent thread 
// 		thread_id = omp_get_thread_num();
// 	}

// 	// Number of bits to generate in each call of the random generator. This is the number of bits in a uint64_t integer.
// 	const int bits_per_byte = 8;
// 	const int num_bits = sizeof(uint64_t) * bits_per_byte;

// 	// Compile each distribution to generate individual row vectors in parallel 
// 	// The parallel section only works if num_threads is non-zero. Otherwise it runs in serial order.
// 	if constexpr(Distr == 0){ // Rademacher distribution 
// 		#pragma omp parallel if (num_threads > 0)
// 		{
// 			thread_id = (num_threads > 0) ? omp_get_thread_num() : 0; // changed

// 			#pragma omp for schedule(static)
// 			for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
// 				std::bitset< 64 > ubits { random_bit_generator.next(thread_id) };
// 				for (IndexType j=0; j < num_bits; ++j) {
// 					array[i*num_bits + j] = ubits[j] ? 1.0 : -1.0; // Shift checks the j-th bit (from right to left) is 1 or 0
// 				}
// 			}
// 		}
// 		// This loop should have less than 64 iterations.
// 		std::bitset< 64 > ubits { random_bit_generator.next(thread_id) };
// 		for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
// 			array[j] = ubits[i] ? 1.0 : -1.0;
// 		}
// 	} else if constexpr(Distr == 1){
// 		return; 
// 	} else if constexpr(Distr == 2){
// 		// rayleigh distribution
// 		std::normal_distribution d{0.0, 1.0};
// 		#pragma omp parallel if (num_threads > 0)
// 		{
// 			thread_id = (num_threads > 0) ? omp_get_thread_num() : 0;

// 			#pragma omp for schedule(static)
// 			for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
// 				for (IndexType j=0; j < num_bits; ++j) {
// 					array[i*num_bits + j] = d(*random_bit_generator.generators[thread_id]);
// 				}
// 			}
// 		}
// 		// This loop should have less than 64 iterations.
// 		for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
// 			array[j] = d(*random_bit_generator.generators[0]);
// 		}
		
// 		DataType sum = 0.0; 
// 		#pragma omp simd for reduction (+:sum)
// 		for (auto i = LongIndexType(0); i < array_size; ++i){
// 			sum = sum + array[i];
// 		}
// 		for (auto i = LongIndexType(0); i < array_size; ++i){
// 			array[i] = array[i] / sum;
// 		}
// 	}
// }   

// template< std::floating_point DataType, ThreadSafeRBG RBG > 
// void generate_array(RBG& rgb, DataType* array, const LongIndexType array_size, const IndexType num_threads, const Distribution d = 0){
// 	// enum Distribution { rademacher, normal, rayleigh };
// 	switch(d){
// 		case rademacher:
// 			generate_array< 0 >(rgb, array, array_size, num_threads);
// 			break;
// 		case normal: 
// 			generate_array< 1 >(rgb, array, array_size, num_threads);
// 			break;
// 		case rayleigh:
// 			generate_array< 2 >(rgb, array, array_size, num_threads);
// 			break;
// 	}
// }

// Simple way to parameterize the random number generator w/ a templated type
// https://stackoverflow.com/questions/5450159/what-type-erasure-techniques-are-there-and-how-do-they-work
// template< int engine_id = 0 >
// auto param_rng(const int seed, const int num_threads = 0) {
//   // "splitmix64", "xoshiro256**", "lcg64", "pcg64", "mt64"
//   if constexpr (engine_id == 0){
//     return ThreadedRNG64< SplitMix64 >(num_threads, seed);
//   } else if constexpr (engine_id == 1){
//     return ThreadedRNG64< Xoshiro256StarStar >(num_threads, seed);
//   } else if constexpr (engine_id == 2){
//     return ThreadedRNG64< knuth_lcg >(num_threads, seed);
//   } else if constexpr (engine_id == 3){
//     return ThreadedRNG64< pcg64 >(num_threads, seed);
//   } else if constexpr (engine_id == 4){
//     return ThreadedRNG64< std::mt19937_64 >(num_threads, seed);
//   } else {
//     throw std::invalid_argument("Invalid random number engine id.");
//   }
// }

// enum rng_engine { sx = 0, xs = 1, lcg = 2, pcg = 3, mt = 4 };
// auto param_rng(const int engine_id, const int seed, const int num_threads = 0){
// 	switch(static_cast< rng_engine >(engine_id)){
// 		case sx: 
// 			return _param_rng< 0 >(seed, num_threads);
// 		default: 
// 			return _param_rng< 1 >(seed, num_threads);
// 	}
// }


// Template declaration
// template < std::floating_point F, ThreadSafeRBG RBG, Distribution dist >
// struct IsotropicSampler;

// template <typename F, typename RBG>
// struct IsotropicSampler< F, RBG, Distribution::rademacher > {
//     using value_type = F;
//     RBG gen;
//     const Distribution dist_id;

//     IsotropicSampler(RBG& rbg) : gen(rbg), dist_id(Distribution::rademacher) {
//         // Additional initialization specific to rademacher distribution
//     }

//     void generate_array(const LongIndexType n, F* output, const IndexType num_threads = 1) {
//         // Implementation for rademacher distribution
//     }
// };

// template <typename F, typename RBG>
// struct IsotropicSampler<F, RBG, Distribution::normal> {
//     using value_type = F;
//     RBG gen;
//     const Distribution dist_id;

//     IsotropicSampler(RBG& rbg) : gen(rbg), dist_id(Distribution::normal) {
//         // Additional initialization specific to normal distribution
//     }

//     void generate_array(const LongIndexType n, F* output, const IndexType num_threads = 1) {
//         // Implementation for normal distribution
//     }
// };

// template <typename F, typename RBG>
// struct IsotropicSampler<F, RBG, Distribution::rayleigh> {
//     using value_type = F;
//     RBG gen;
//     const Distribution dist_id;

//     IsotropicSampler(RBG& rbg) : gen(rbg), dist_id(Distribution::rayleigh) {
//         // Additional initialization specific to rayleigh distribution
//     }

//     void generate_array(const LongIndexType n, F* output, const IndexType num_threads = 1) {
//         // Implementation for rayleigh distribution
//     }
// };



// template< std::floating_point F, ThreadSafeRBG RBG >
// struct IsotropicSampler {
// 	using value_type = F; 
// 	RBG gen;
// 	const Distribution dist_id; 
// 	IsotropicSampler(RBG& rbg, const Distribution d) : gen(rbg), dist_id(d) {
// 		return;	
// 	}
// 	void generate(const LongIndexType n, F* output, const IndexType num_threads = 1){
// 		switch(dist_id){
// 			case rademacher:
// 		}
// 		generate_array< 0, F, RBG >(this->gen, output, n, num_threads);
// 	};
// };

// Since temlate specialization is just wayy too hard, we just use if constexpr and move on
// Rademacher distribution
// template< >
// void IsotropicSampler< F, RBG >::generate_array< Distribution::rademacher >(
// 	const LongIndexType n, 
// 	F* output, 
// 	const IndexType num_threads = 1
// ){
// 	#pragma omp parallel if (num_threads > 0)
// 	{
// 		thread_id = (num_threads > 0) ? omp_get_thread_num() : 0; // changed
// 		#pragma omp for schedule(static)
// 		for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
// 			std::bitset< 64 > ubits { random_bit_generator.next(thread_id) };
// 			for (IndexType j=0; j < num_bits; ++j) {
// 				array[i*num_bits + j] = ubits[j] ? 1.0 : -1.0; //Shift checks the j-th bit (from right to left) is 1 or 0
// 			}
// 		}
// 	}
// 	// This loop should have less than 64 iterations.
// 	std::bitset< 64 > ubits { random_bit_generator.next(thread_id) };
// 	for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
// 		array[j] = ubits[i] ? 1.0 : -1.0;
// 	}
// }

// // Normal distribution 
// template<>
// void IsotropicSampler::generate_array< Distribution::normal > () {
// 	// Zero mean, unit variance gaussian
// 	std::normal_distribution d {0.0, 1.0};
// 	#pragma omp parallel if (num_threads > 0)
// 	{
// 		thread_id = (num_threads > 0) ? omp_get_thread_num() : 0;
// 		#pragma omp for schedule(static)
// 		for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
// 			for (IndexType j=0; j < num_bits; ++j) {
// 				array[i*num_bits + j] = d(random_bit_generator.generators[thread_id]);
// 			}
// 		}
// 	}
// 	// This loop should have less than 64 iterations.
// 	for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
// 		array[j] = d(random_bit_generator.generators[0]);
// 	}
// }

// // rayleigh distribution
// template<>
// void IsotropicSampler::generate_array< Distribution::rayleigh > () {
// 	std::normal_distribution d{0.0, 1.0};
// 	#pragma omp parallel if (num_threads > 0)
// 	{
// 		thread_id = (num_threads > 0) ? omp_get_thread_num() : 0;

// 		#pragma omp for schedule(static)
// 		for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
// 			for (IndexType j=0; j < num_bits; ++j) {
// 				array[i*num_bits + j] = d(random_bit_generator.generators[thread_id]);
// 			}
// 		}
// 	}
// 	// This loop should have less than 64 iterations.
// 	for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
// 		array[j] = d(random_bit_generator.generators[0]);
// 	}
	
// 	DataType sum = 0.0; 
// 	#pragma omp parallel for reduction (+:sum)
// 	for (auto i = LongIndexType(0); i < array_size; ++i){
// 		sum = sum + array[i];
// 	}
// 	for (auto i = LongIndexType(0); i < array_size; ++i){
// 		array[i] = array[i] / sum;
// 	}
// }

#endif 