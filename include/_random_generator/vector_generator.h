#ifndef _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_

#include <omp.h>   // omp_set_num_threads, omp_get_thread_num
#include <random>   // uniform_random_bit_generator, normal_distribution
#include <cstdint>  // uint64_t
#include <bitset>   // bitset
#include "random_concepts.h" // ThreadSafeRBG

using IndexType = unsigned int;
using LongIndexType = unsigned long;
static constexpr long num_bits = 64; 

// TODO: make isotropic generator class that accepts random-bit genertaor
enum Distribution { rademacher = 0, normal = 1, rayleigh = 2 };

// Generates an isotropic random vector for a given thread id
template< size_t Distr, std::floating_point F, ThreadSafeRBG RBG > 
void generate_isotropic(RBG& random_bit_generator, F* array, const long n, const int thread_id){
	const auto N = static_cast< unsigned long >(n / RBG::num_bits);
	for (auto i = 0; i < N; ++i) {
		std::bitset< RBG::num_bits > ubits { random_bit_generator.next(thread_id) };
		
		#pragma omp simd
		for (size_t j = 0; j < RBG::num_bits; ++j) {
			// array[i*RBG::num_bits + j] = ubits[j] ? 1.0 : -1.0; // Shift checks the j-th bit (from right to left) is 1 or 0
			array[i*RBG::num_bits + j] = 2 * int(ubits[j]) - 1;
		}
	}

	// This loop should have less than 64 iterations.
	std::bitset< RBG::num_bits > ubits { random_bit_generator.next(thread_id) };
	for (size_t j = N * RBG::num_bits, i = 0; j < n; ++j, ++i){
		// array[j] = ubits[i] ? 1.0 : -1.0;
		array[j] = 2 * int(ubits[j]) - 1;
	}
}

template< size_t Distr, std::floating_point DataType, ThreadSafeRBG RBG > 
void generate_array(
	RBG& random_bit_generator,
	DataType* array,
	const LongIndexType array_size,
	const IndexType num_threads
){
	// Passing zero-threads indicates caller was already parallelized and thus should not re-set threads
	// Otherwise 'num_threads' threads is created 
	if (num_threads > 0) {
		omp_set_num_threads(num_threads);
	}

	// Get the thread id
	int thread_id = 0;
	if (num_threads == 0) { // if in parent thread 
		thread_id = omp_get_thread_num();
	}

	// Number of bits to generate in each call of the random generator. This is the number of bits in a uint64_t integer.
	const int bits_per_byte = 8;
	const int num_bits = sizeof(uint64_t) * bits_per_byte;

	// Compile each distribution to generate individual row vectors in parallel 
	// The parallel section only works if num_threads is non-zero. Otherwise it runs in serial order.
	if constexpr(Distr == 0){ // Rademacher distribution 
		#pragma omp parallel if (num_threads > 0)
		{
			thread_id = (num_threads > 0) ? omp_get_thread_num() : 0; // changed

			#pragma omp for schedule(static)
			for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
				std::bitset< 64 > ubits { random_bit_generator.next(thread_id) };
				for (IndexType j=0; j < num_bits; ++j) {
					array[i*num_bits + j] = ubits[j] ? 1.0 : -1.0; // Shift checks the j-th bit (from right to left) is 1 or 0
				}
			}
		}
		// This loop should have less than 64 iterations.
		std::bitset< 64 > ubits { random_bit_generator.next(thread_id) };
		for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
			array[j] = ubits[i] ? 1.0 : -1.0;
		}
	} else if constexpr(Distr == 1){
		// Zero mean, unit variance gaussian
		std::normal_distribution d {0.0, 1.0};
		#pragma omp parallel if (num_threads > 0)
		{
			thread_id = (num_threads > 0) ? omp_get_thread_num() : 0;

			#pragma omp for schedule(static)
			for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
				for (IndexType j=0; j < num_bits; ++j) {
					array[i*num_bits + j] = d(random_bit_generator.generators[thread_id]);
				}
			}
		}
		// This loop should have less than 64 iterations.
		for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
			array[j] = d(random_bit_generator.generators[0]);
		}
	} else if constexpr(Distr == 2){
		// rayleigh distribution
		std::normal_distribution d{0.0, 1.0};
		#pragma omp parallel if (num_threads > 0)
		{
			thread_id = (num_threads > 0) ? omp_get_thread_num() : 0;

			#pragma omp for schedule(static)
			for (LongIndexType i=0; i < static_cast<LongIndexType>(array_size/num_bits); ++i) {
				for (IndexType j=0; j < num_bits; ++j) {
					array[i*num_bits + j] = d(random_bit_generator.generators[thread_id]);
				}
			}
		}
		// This loop should have less than 64 iterations.
		for (auto j = LongIndexType(array_size/num_bits) * num_bits, i = LongIndexType(0); j < array_size; ++j, ++i){
			array[j] = d(random_bit_generator.generators[0]);
		}
		
		DataType sum = 0.0; 
		#pragma omp parallel for reduction (+:sum)
		for (auto i = LongIndexType(0); i < array_size; ++i){
			sum = sum + array[i];
		}
		for (auto i = LongIndexType(0); i < array_size; ++i){
			array[i] = array[i] / sum;
		}
	}
}   

template< std::floating_point DataType, ThreadSafeRBG RBG > 
void generate_array(RBG& random_bit_generator, DataType* array, const LongIndexType array_size, const IndexType num_threads, const Distribution d){
	// enum Distribution { rademacher, normal, rayleigh };
	switch(d){
		case rademacher:
			generate_array< 0 >(random_bit_generator, array, array_size, num_threads);
			break;
		case normal: 
			generate_array< 1 >(random_bit_generator, array, array_size, num_threads);
			break;
		case rayleigh:
			generate_array< 2 >(random_bit_generator, array, array_size, num_threads);
			break;
	}
}


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