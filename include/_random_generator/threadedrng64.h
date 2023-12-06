#ifndef _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_


#include <stdint.h>  // uint64_t
#include <cassert>  // assert
#include <cstdlib>  // NULL
#include <random>   // uniform_random_bit_generator, mt19937_64
#include <functional> // std::function 
#include "random_concepts.h" // LightRandom64Engine, Random64Engine
#include "rne_engines.h" // all the engines
// #include "./xoshiro_256_star_star.h"  // Xoshiro256StarStar
// #include "./pcg_random.h" // PCG 

enum RbEngine { sx = 0, xs = 1, pcg = 2, lcg = 3, mt = 4 };

// Thread-safe random number generator
// This class constructs n_thread copies of a given *random number engine* type (a state machine with a transition + output function) using 
// different seed sequences (seed_seq's) generated from the std::random_device, such that one can safely call ThreadedRNG64.next(tid) with the 
// give thread id (tid) and obtain a uniformly random unsigned integer with at least 64-bits. 
// NOTE: RandomNumberEngine == UniformRandomBitGenerator + has .seed(), default constructible, and other things
// template< LightRandom64Engine RNE = std::mt19937_64 >
struct ThreadedRNG64 {
	static constexpr size_t num_bits = 64;
	int num_threads;
	const RbEngine engine_id = sx; 
	std::vector< Random64EngineConcept* > generators;
	ThreadedRNG64(int engine = 2) : engine_id(static_cast< RbEngine >(engine)) {
		// std::uniform_random_bit_generator RBG = std::random_device;
		int num_threads_ = 1;
		initialize(num_threads_);
	};
	explicit ThreadedRNG64(int num_threads_, int engine, int seed = -1) : engine_id(static_cast< RbEngine >(engine)) {
		initialize(num_threads_, seed);
	};
	~ThreadedRNG64(){
		for (int i = 0; i < num_threads; ++i){
			// delete generators[i]; // todo: see if this is causing heap corruption
		}
	}
	auto next(int thread_id) -> std::uint64_t {
		return generators[thread_id]->operator()();
	}

	void initialize(int num_threads_, int seed = -1){
		// assert(num_threads_ > 0);
		if (num_threads_ == 0){ return; }
		num_threads = num_threads_;
		generators = std::vector< Random64EngineConcept* >(num_threads, nullptr);
		
		// Make new generators
		for (size_t i = 0; i < generators.size(); ++i){
			switch(engine_id){
				case sx:
					static_assert(std::uniform_random_bit_generator< Random64Engine< SplitMix64 > >, "Wrapper RNG engine constraints not met");
					generators[i] = new Random64Engine< SplitMix64 >();
					break; 
				case xs: 
					generators[i] = new Random64Engine< Xoshiro256StarStar >();
					break; 
				case pcg: 
					generators[i] = new Random64Engine< pcg64 >();
					break; 
				case lcg:
					generators[i] = new Random64Engine< knuth_lcg >();
					break; 
				case mt: 
					generators[i] = new Random64Engine< std::mt19937_64 >();
					break; 
			}
		}

		// Seeds generators with sources of entropy from a RBG (e.g. random_device)
		// This seeds the entire state vector of the corresponding RNE's state size / entropy source using rd
		auto rdev = std::random_device();
		auto mt = std::mt19937(seed);
		std::function< std::uint_fast32_t() > rd;
		if (seed == -1){
			rd = [&rdev](){ return rdev(); };
		} else {
			rd = [&mt](){ return mt(); };
		} 

		// Saturate the full state of the generators with entropy
		for (int i = 0; i < num_threads; ++i) {
			const auto ssize = generators[i]->state_size(); 
			// if (ssize > 0){
			std::vector< uint32_t > seed_data(ssize, 0);
			std::generate_n(seed_data.begin(), ssize, rd); // generate evenly-distributed 32-bit seeds
			std::seed_seq seed_gen(std::begin(seed_data), std::end(seed_data));
			generators[i]->seed(seed_gen);
			// } else {
			// 	uint64_t seed = (uint64_t(rd()) << 32) | rd();
			// 	generators[i]->seed(seed);
			// }
		}
	};
};


#endif  // _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
