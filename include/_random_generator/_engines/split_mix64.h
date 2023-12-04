#ifndef _RANDOM_GENERATOR_SPLIT_MIX_64_H_
#define _RANDOM_GENERATOR_SPLIT_MIX_64_H_

#include <stdint.h>  // uint64_t
#include <cassert>  // assert
#include <ctime>  // std::time
#include <random> // std::seed_seq
#include <array> // array 

/// \class SplitMix64
///
/// \brief   Pseudo-random integer generator. This class generates 64-bit
///          integer using SplitMix64 algorithm.
///
/// \details The SplitMix64 algorithm is very fast but does not pass all
///          statistical tests. This class is primarily used to initialize the
///          states of the \c Xoshiro256StarStar class.
///
///          The SplitMix64 algorithm is develped by Sebastiano Vigna (2015)
///          and the source code is available at:
///          https://prng.di.unimi.it/splitmix64.c
///
/// \sa      Xoshiro256StarStar

struct SplitMix64 {
    // static constexpr typename result_type = uint64_t; 
    using result_type = uint64_t;
    static constexpr size_t state_size = 2; // size of the entropy/state size, in units of 32-bits
    uint64_t state;
    SplitMix64(){
        uint64_t seed = static_cast<uint64_t>(std::time(0)) + static_cast<uint64_t>(std::clock());
        state = (seed << 32) | seed;
    };
    void seed(std::seed_seq& state_) {
        std::array< std::uint32_t, 2 > seeds = { 0, 0 }; 
        state_.generate(seeds.begin(), seeds.end());
        state = static_cast< std::uint64_t >(seeds[0]);
        state = state << 32; 
        state = state | static_cast< std::uint64_t >(seeds[1]);
    };
    uint64_t operator()() noexcept {
        uint64_t z = (state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    };
    static constexpr uint64_t min(){ return std::numeric_limits< uint64_t >::min(); }
    static constexpr uint64_t max(){ return std::numeric_limits< uint64_t >::max(); }
};

#endif  // _RANDOM_GENERATOR_SPLIT_MIX_64_H_
