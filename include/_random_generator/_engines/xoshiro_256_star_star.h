/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>, Matt Piekenbrock <matt.piekenbrock@gmail.com>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_XOSHIRO_256_STAR_STAR_H_
#define _RANDOM_GENERATOR_XOSHIRO_256_STAR_STAR_H_


#include <array> 
#include <stdint.h>  // uint64_t, UINT64_C
#include <cstdlib>  // NULL
#include "./split_mix64.h"  // SplitMix64

// stdint.h in old compilers (e.g. gcc 4.4.7) does not declare UINT64_C macro.
#ifndef UINT64_C
    #define UINT64_C(c) static_cast<uint64_t>(c)
#endif

// =====================
// Xoshiro 256 Star Star
// =====================

/// \class   Xoshiro256StarStar
///
/// \brief   Pseudo-random integer generator. This class generates 64-bit
///          integer using Xoshiro256** algorithm.
///
/// \details The Xoshiro256** algorithm has 256-bit state space, and passes all
///          statistical tests, including the BigCrush. The state of this class
///          is initialized using \c SplitMix64 random generator.
///
///          A very similar method to Xoshiro256** is Xoshiro256++ which has
///          the very same properties and speed as the Xoshiro256**. An
///          alternative method is Xoshiro256+, which is 15% faster, but it
///          suffers linear dependency of the lower 4 bits. It is usually used
///          for generating floating numbers using the upper 53 bits and
///          discard the lower bits.
///
///          The Xoshiro256** algorithm is develped by David Blackman and
///          Sebastiano Vigna (2018) and the source code can be found at:
///          https://prng.di.unimi.it/xoshiro256starstar.c
///
/// \sa      SplitMix64


struct Xoshiro256StarStar{
    using result_type = uint64_t;
    static constexpr size_t state_size = 8;
    static inline uint64_t rotation_left( const uint64_t x, int k){
        return (x << k) | (x >> (64 - k));
    };
    std::array< uint64_t, 4 > state;

    Xoshiro256StarStar(){
        SplitMix64 split_mix_64;
        state = { split_mix_64(), split_mix_64(), split_mix_64(), split_mix_64() };
    };
    void seed(std::seed_seq& S){
        std::array< std::uint32_t, 8 > seeds;
        S.generate(seeds.begin(), seeds.end());
        state[0] = (static_cast< std::uint64_t >(seeds[0]) << 32) | seeds[1];
        state[1] = (static_cast< std::uint64_t >(seeds[2]) << 32) | seeds[3];
        state[2] = (static_cast< std::uint64_t >(seeds[4]) << 32) | seeds[5];
        state[3] = (static_cast< std::uint64_t >(seeds[6]) << 32) | seeds[7];
    };
    uint64_t operator()(){
        const uint64_t result = rotation_left(state[1] * 5, 7) * 9;
        const uint64_t t = state[1] << 17;  
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= t;
        state[3] = rotation_left(state[3], 45);
        return result;
    };

    void jump(){
        static const uint64_t JUMP[] = {
            0x180ec6d33cfd0aba,
            0xd5a61266f0c9392c,
            0xa9582618e03fc9aa,
            0x39abdc4529b1661c
        };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;

        for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); ++i) {
            for (int b = 0; b < 64; ++b) {
                if (JUMP[i] & UINT64_C(1) << b) {
                    s0 ^= state[0];
                    s1 ^= state[1];
                    s2 ^= state[2];
                    s3 ^= state[3];
                }
                this->operator()();
            }
        }

        state[0] = s0;
        state[1] = s1;
        state[2] = s2;
        state[3] = s3;
    };
    void long_jump(){
        static const uint64_t LONG_JUMP[] = {
            0x76e15d3efefdcbbf,
            0xc5004e441c522fb3,
            0x77710069854ee241,
            0x39109bb02acbe635
        };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;

        for (unsigned int i = 0; i < sizeof(LONG_JUMP) / sizeof(*LONG_JUMP); ++i) {
            for (int b = 0; b < 64; ++b) {
                if (LONG_JUMP[i] & UINT64_C(1) << b) {
                    s0 ^= state[0];
                    s1 ^= state[1];
                    s2 ^= state[2];
                    s3 ^= state[3];
                }
                this->operator()();
            }
        }
        state[0] = s0;
        state[1] = s1;
        state[2] = s2;
        state[3] = s3;
    };
    static constexpr uint64_t min(){ return std::numeric_limits< uint64_t >::min(); }
    static constexpr uint64_t max(){ return std::numeric_limits< uint64_t >::max(); }
};


#endif  // _RANDOM_GENERATOR_XOSHIRO_256_STAR_STAR_H_
