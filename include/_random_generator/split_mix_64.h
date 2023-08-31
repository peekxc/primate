/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_SPLIT_MIX_64_H_
#define _RANDOM_GENERATOR_SPLIT_MIX_64_H_


// ======
// Header
// ======

#include <stdint.h>  // uint64_t
#include <cassert>  // assert
#include <ctime>  // std::time

// ============
// Split Mix 64
// ============

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

struct SplitMix64
{
        SplitMix64(){
            // std::time gives the second since epoch. This, if this function is called
            // multiple times a second, the std::time() results the same number. To
            // make it differ between each milliseconds, the std::clock is added, which
            // is the cpu time (in POSIX) or wall time (in windows) and in the unit of
            // system's clocks per second.
            uint64_t seed = static_cast<uint64_t>(std::time(0)) +
                            static_cast<uint64_t>(std::clock());

            // Seeding as follow only fills the first 32 bits of the 64-bit integer.
            // Repeat the first 32 bits on the second 32-bits to create a better 64-bit
            // random number
            this->state = (seed << 32) | seed;
        };
        explicit SplitMix64(uint64_t state_) : state(state_) {
            // Initial state must not be zero.
            assert(state_ != 0);
        };
        uint64_t next(){
            uint64_t z = (state += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;

            return z ^ (z >> 31);
        };

    protected:
        uint64_t state;
};

#endif  // _RANDOM_GENERATOR_SPLIT_MIX_64_H_
