#ifndef _RANDOM_GENERATOR_RNE_ENGINES_H
#define _RANDOM_GENERATOR_RNE_ENGINES_H

#include <random>  // mt
#include "_engines/pcg_random.h" // pcg64 -- this adds a non-trivial amount to compile time
#include "_engines/split_mix64.h"  // SplitMix64
#include "_engines/xoshiro_256_star_star.h" // Xoshiro256

// From: https://www.pcg-random.org/posts/cpp-seeding-surprises.html
// This is technically a biased generator, but with 64-bits of entropy it may be ok 
// for many applications and it should be blazing fast besides
using knuth_lcg = std::linear_congruential_engine< uint64_t, 6364136223846793005U, 1442695040888963407U, 0U>;


#endif 