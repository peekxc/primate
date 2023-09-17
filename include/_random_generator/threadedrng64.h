/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_


#include <stdint.h>  // uint64_t
#include <cassert>  // assert
#include <cstdlib>  // NULL
#include <random>   // uniform_random_bit_generator, mt19937_64
#include "random_concepts.h" // LightRandom64Engine
// #include "./xoshiro_256_star_star.h"  // Xoshiro256StarStar
// #include "./pcg_random.h" // PCG 

// Thread-safe random bit generator
// This class constructs n_thread copies of a given *random number engine* type (a state machine with a transition + output function) using 
// different seed sequences (seed_seq's) generated from the std::random_device, such that one can safely call ThreadedRNG64.next(tid) with the 
// give thread id (tid) and obtain a uniformly random unsigned integer with at least 64-bits. 
// NOTE: RandomNumberEngine == UniformRandomBitGenerator + has .seed(), default constructible, and other things
template< LightRandom64Engine RNE = std::mt19937_64, std::uniform_random_bit_generator RBG = std::random_device >
struct ThreadedRNG64 {
    int num_threads;
    std::vector< RNE > generators;
    ThreadedRNG64(){
        int num_threads_ = 1;
        initialize(num_threads_);
    };
    explicit ThreadedRNG64(int num_threads_){
        initialize(num_threads_);
    };
    auto next(int thread_id) -> std::uint_fast64_t {
        return generators[thread_id]();
    }

    void initialize(int num_threads_){
        assert(num_threads_ > 0);
        num_threads = num_threads_;
        generators = std::vector< RNE >(num_threads);

        // Seeds generators with sources of entropy from a RBG (e.g. random_device)
        // This seeds the entire state vector of the corresponding RNE's state size / entropy source using rd
        RBG rd;
        if constexpr(Random64Engine< RNE >){
            std::uint_fast32_t seed_data[RNE::state_size];
            for (int i = 0; i < num_threads; ++i) {
                std::generate_n(seed_data, RNE::state_size, std::ref(rd)); // generate evenly-distributed 32-bit seeds
                std::seed_seq seed_gen(std::begin(seed_data), std::end(seed_data));
                generators[i].seed(seed_gen);
            }
        } else {
            for (int i = 0; i < num_threads; ++i) {
                uint64_t seed = (uint64_t(rd()) << 32) | rd();
                generators[i].seed(seed);
            }
        }
        
    };
};


#endif  // _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
