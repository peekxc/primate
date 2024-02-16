// PCG64.h
// Adapted from C-implementation: (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
#ifndef _RANDOM_GENERATOR_PCG64_H_
#define _RANDOM_GENERATOR_PCG64_H_

#include <cstdint>  // uint64_t
#include <cassert>  // assert
#include <ctime>    // std::time
#include <random>   // std::seed_seq
#include <array>    // array

struct Pcg64 { 
  using result_type = uint64_t;
  static constexpr size_t state_size = 2; // size of the entropy/state size, in units of 32-bits
  uint64_t state;  
  uint64_t inc; 

  Pcg64(){
    uint64_t seed = static_cast<uint64_t>(std::time(0)) + static_cast<uint64_t>(std::clock());
    state = (seed << 32) | seed;
    inc = (state << 1u) | 1u;
  };

  void seed(std::seed_seq& seed_s) {
    std::array< std::uint32_t, 2 > seeds = { 0, 0 }; 
    seed_s.generate(seeds.begin(), seeds.end());
    state = static_cast< std::uint64_t >(seeds[0]);
    state = state << 32; 
    state = state | static_cast< std::uint64_t >(seeds[1]);
    inc = (state << 1u) | 1u;
  };
  
  uint32_t _generate32() noexcept {
    uint64_t oldstate = state;
    // Advance internal state
    state = oldstate * 6364136223846793005ULL + (inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  };

  uint64_t operator()() noexcept {
    uint64_t z = _generate32();
    z = z << 32; 
    z = z | static_cast< std::uint64_t >(_generate32());
    return z;
  }

  static constexpr uint64_t min(){ return std::numeric_limits< uint64_t >::min(); }
  static constexpr uint64_t max(){ return std::numeric_limits< uint64_t >::max(); }
};

#endif  // _RANDOM_GENERATOR_PCG64_H_
