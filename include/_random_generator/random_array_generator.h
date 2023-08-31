/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_


// =======
// Headers
// =======
#include "../_definitions/types.h"  // IndexType, LongIndexType
#include "./random_number_generator.h"
#include <omp.h>  // omp_set_num_threads, omp_get_thread_num
#include <stdint.h>  // uint64_t
// #include "../_c_trace_estimator/c_orthogonalization.h"  // cOrthogonalization
// #include "../_c_basic_algebra/c_vector_operations.h"  // cVectorOperations

template <typename DataType>
struct RandomArrayGenerator
{
// =====================
// generate random array
// =====================

/// \brief      Generates a pseudo-random array with Rademacher distribution
///             where elements are either \c +1 or \c -1.
///
/// \details    The Rademacher distribution is obtained from the Bernoulli
///             distribution consisting of \c 0 and \c 1. To generate such
///             distribution, a sequence of \c array_size/64 intergers, each
///             with 64-bit, is generated using Xoshiro256** algorithm. The 64
///             bits of each integer are used to fill 64 elements of the array
///             as follows. If the bit is \c 0, the array element is set to \c
///             -1, and if the bit is \c 1, the array element is set to \c +1.
///
///             Thus, in this function, we use Xoshiro256** algorithm to
///             generate 64 bits and use bits, not the integer itself. This
///             approach is about ten times faster than convertng the random
///             integer to double between \c [0,1] and then map them to \c +1
///             and \c -1.
///
///             Also, this function is more than a hundered times faster than
///             using \c rand() function.
///
/// \param[in]  random_number_generator
///             The random number generator object. This object should be
///             initialized with \c num_threads by its constructor. On each
///             parallel thread, an independent sequence of random numbers are
///             generated.
/// \param[out] array
///             1D array of the size \c array_size.
/// \param[out] array_size
///             The size of the array.
/// \param[in]  num_threads
///             Number of OpenMP parallel threads. If \c num_threads is zero
///             then no paralel thread is created inside this function, rather
///             it is assumed that this functon is called inside a parallel
///             region from the caller.

static void generate_random_array(
        RandomNumberGenerator& random_number_generator,
        DataType* array,
        const LongIndexType array_size,
        const IndexType num_threads)
{
    // Set the number of threads only if num_threads is non-zero. If
    // num_threads is zero, it indicates to not create new threads in this
    // function, rather, this function was called from another caller function
    // that was already parallelized and this function should be executed in
    // one of those threads.
    if (num_threads > 0)
    {
        // num_threads parallel threads will be created in this function.
        omp_set_num_threads(num_threads);
    }

    // Finding the thread id. It depends where we call omp_get_thread_num. If
    // we call it here (below), it gives the thread id of the parent function.
    // But if we call it inside a #pragma omp parallel, it gives the newer
    // thread id that is "created" by the pragma.
    int thread_id = 0;
    if (num_threads == 0)
    {
        // If num_threads is zero (which means we will not create a new thread
        // in this function), we get the thread id of the parent function.
        thread_id = omp_get_thread_num();
    }

    // Number of bits to generate in each call of the random generator. This is
    // the number of bits in a uint64_t integer.
    const int bits_per_byte = 8;
    const int num_bits = sizeof(uint64_t) * bits_per_byte;

    // Shared-memory parallelism over individual row vectors. The parallel
    // section only works if num_threads is non-zero. Otherwise it runs in
    // serial order.
    #pragma omp parallel if (num_threads > 0)
    {
        // If num_threads is zero, the following thread_id is the thread id
        // that the parent (caller) function created outside of this function.
        // But, if num_thread is non-zero, the following thread id is the
        // thread id that is created inside this parallel loop.
        if (num_threads > 0)
        {
            thread_id = omp_get_thread_num();
        }

        #pragma omp for schedule(static)
        for (LongIndexType i=0;
             i < static_cast<LongIndexType>(array_size/num_bits); ++i)
        {
            // Generate 64 bits (one integer)
            uint64_t bits = random_number_generator.next(thread_id);

            // Fill 64 elements of array with +1 or -1 depending on the bits
            for (IndexType j=0; j < num_bits; ++j)
            {
                // Check if the j-th bit (from right to left) is 1 or 0
                if (bits & ( uint64_t(1) << j))
                {
                    // Bit is 1. Write +1.0 in array
                    array[i*num_bits + j] = 1.0;
                }
                else
                {
                    // Bit is 0. Write -1.0 in array
                    array[i*num_bits + j] = -1.0;
                }
            }
        }
    }

    // The previous for loop (the above) does not fill all elements of array
    // since it only iterates on the multiples of 64 (num_bits). We fill the
    // rest of the array. There are less than 64 elements remained to fill. So,
    // it suffice to generate only 64 bits (one integer) for the rest of the
    // elements of the array
    uint64_t bits = random_number_generator.next(thread_id);

    // This loop should have less than 64 iterations.
    for (LongIndexType j = \
            static_cast<LongIndexType>(array_size/num_bits) * num_bits;
         j < array_size; ++j)
    {
        // Check if the j-th bit (from right to left) is 1 or 0
        if (bits & ( uint64_t(1) << j))
        {
            // Bit is 1. Write +1.0 in array
            array[j] = 1.0;
        }
        else
        {
            // Bit is 0. Write -1.0 in array
            array[j] = -1.0;
        }
    }
};
};

// ===============================
// Explicit template instantiation
// ===============================

template struct RandomArrayGenerator<float>;
template struct RandomArrayGenerator<double>;
template struct RandomArrayGenerator<long double>;

#endif 