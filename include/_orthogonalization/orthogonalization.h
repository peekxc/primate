/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _ORTHOGONALIZATION_H_
#define _ORTHOGONALIZATION_H_

// =======
// Imports
// =======

#include <cstdlib>  // abort
#include <iostream>  // std::cerr, std::endl
#include <cmath>  // sqrt, std::fabs
#include <limits>  // std::numeric_limits
#include "../_c_basic_algebra/c_vector_operations.h"  
#include "../_random_generator/random_concepts.h" 
#include "../_random_generator/threadedrng64.h" 
#include "../_random_generator/vector_generator.h"


template <typename DataType>
struct cOrthogonalization {

// ====================
// gram schmidt process
// ====================

/// \brief         Modified Gram-Schmidt orthogonalization process to
///                orthogonalize the vector \c v against a subset of the column
///                vectors in the array \c V.
///
/// \details       \c V is 1D array of the length \c vector_size*num_vectors to
///                represent a 2D array of a set of \c num_vectors column
///                vectors, each of the length \c vector_size. The length of
///                \c v is also \c vector_size.
///
///                \c v is orthogonalized against the last \c num_ortho
///                columns of \c V starting from the column vector of the index
///                \c last_vector. If the backward indexing from \c last_vector
///                becomes a negative index, the index wraps around from the
///                last column vector index, i.e., \c num_vectors-1 .
///
///                * If \c num_ortho is zero, or if \c num_vectors is zero, no
///                  orthogonalization is performed.
///                * If \c num_ortho is negative (usually set to \c -1), then
///                  \c v is orthogonalized against all column vectors of \c V.
///                * If \c num_ortho is larger than \c num_vectors, then \c v
///                  is orthogonalized against all column vectors of \c V.
///                * If \c num_ortho is smaller than \c num_vectors, then
///                  \c v is orthogonalized against the last \c num_ortho
///                  column vectors of \c V, starting from the column vector
///                  with the index \c last_vector toward its previous vectors.
///                  If the iteration runs into negativen column indices, the
///                  column indexing wraps around from the end of the columns
///                  from \c num_vectors-1.
///
///                The result of the newer \c v is written in-place in \c v.
///
///                If vector \c v is identical to one of the vectors in \c V,
///                the orthogonalization against the identical vector is
///                skipped.
///
///                If one of the column vectors of \c V is zero (have zero
///                norm), that vector is ignored.
///
/// \note          It is assumed that the caller function fills the column
///                vectors of \c V periodically in a *wrapped around* order
///                from column index \c 0,1,... to \c num_vectors-1, and newer
///                vectors are replaced on the wrapped index starting from
///                index \c 0,1,... again. Thus, \c V only stores the last
///                \c num_vectors column vectors. The index of the last filled
///                vector is indicated by \c last_vector.
///
/// \warning       The vector \c v can be indeed one of the columns of \c V
///                itself. However, in this case, vector \c v must *NOT* be
///                orthogonalized against itself, rather, it should only be
///                orthogonalized to the other vectors in \c V. For instance,
///                if \c num_vectors=10, and \c v is the 3rd vector of \c V,
///                and if \c num_ortho is \c 6, then we may set
///                \c last_vector=2. Then \c v is orthogonalized againts the
///                six columns \c 2,1,0,9,8,7, where the last three of them are
///                wrapped around the end of the columns.
///
/// \sa            c_golub_kahn_bidiagonalizaton,
///                c_lanczos_bidiagonalization
///
/// \param[in]     V
///                1D coalesced array of vectors representing a 2D array. The
///                length of this 1D array is \c vector_size*num_vectors, which
///                indicates a 2D array with the shape
///                \c (vector_size,num_vectors).
/// \param[in]     vector_size
///                The length of each vector. If we assume \c V indicates a 2D
///                vector, this is the number of rows of \c V.
/// \param[in]     num_vectors
///                The number of column vectors. If we assume \c V indicates a
///                2D vector, this the number of columns of \c V.
/// \param[in]     last_vector
///                The column vectors of the array \c V are rewritten by the
///                caller function in wrapped-around order. That is, once all
///                the columns (from the zeroth to the \c num_vector-1 vector)
///                are filled, the next vector is rewritten in the place of
///                the zeroth vector, and the indices of newer vectors wrap
///                around the columns of \c V. Thus, \c V only retains the last
///                \c num_vectors vectors. The column index of the last written
///                vector is given by \c last_vector. This index is a number
///                between \c 0 and \c num_vectors-1. The index of the last
///                i-th vector is winding back from the last vector by
///                <tt>last_vector-i+1 mod num_vectors</tt>.
/// \param[in]     num_ortho
///                The number of vectors to be orthogonalized starting from the
///                last vector. \c 0 indicates no orthogonalization will be
///                performed and the function just returns. A negative value
///                means all vectors will be orthogonalized. A poisitive value
///                will orthogonalize the given number of vectors. This value
///                cannot be larger than the number of vectors.
/// \param[in,out] v
///                The vector that will be orthogonalized against the columns
///                of \c V. The length of \c v is \c vector_size. This vector
///                is modified in-place.

static void gram_schmidt_process(
        const DataType* V,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType last_vector,
        const FlagType num_ortho,
        DataType* v)
{
    // Determine how many previous vectors to orthogonalize against
    IndexType num_steps;
    if ((num_ortho == 0) || (num_vectors < 2))
    {
        // No orthogonalization is performed
        return;
    }
    else if ((num_ortho < 0) ||
             (num_ortho > static_cast<FlagType>(num_vectors)))
    {
        // Orthogonalize against all vectors
        num_steps = num_vectors;
    }
    else
    {
        // Orthogonalize against only the last num_ortho vectors
        num_steps = num_ortho;
    }

    // Vectors can be orthogonalized at most to the full basis of the vector
    // space. Thus, num_steps cannot be larger than the dimension of vector
    // space, which is vector_size.
    if (num_steps > static_cast<IndexType>(vector_size))
    {
        num_steps = vector_size;
    }

    IndexType i;
    DataType inner_prod;
    DataType norm;
    DataType norm_v;
    DataType epsilon = std::numeric_limits<DataType>::epsilon();
    DataType distance;

    // Iterate over vectors
    for (IndexType step=0; step < num_steps; ++step)
    {
        // i is the index of a column vector in V to orthogonalize v against it
        if ((last_vector % num_vectors) >= step)
        {
            i = (last_vector % num_vectors) - step;
        }
        else
        {
            // Wrap around negative indices from the end of column index
            i = (last_vector % num_vectors) - step + num_vectors;
        }

        // Norm of j-th vector
        norm = cVectorOperations<DataType>::euclidean_norm(
                &V[vector_size*i], vector_size);

        // Check norm
        if (norm < epsilon * sqrt(vector_size))
        {
            std::cerr << "WARNING: norm of the given vector is too small. " \
                      << "Cannot orthogonalize against zero vector. " \
                      << "Skipping." << std::endl;
            continue;
        }

        // Projection
        inner_prod = cVectorOperations<DataType>::inner_product(
                &V[vector_size*i], v, vector_size);

        // scale for subtraction
        DataType scale = inner_prod / (norm * norm);

        // If scale is is 1, it is possible that vector v and j-th vector are
        // identical (or close).
        if (std::abs(scale - 1.0) <= 2.0 * epsilon)
        {
            // Norm of the vector v
            norm_v = cVectorOperations<DataType>::euclidean_norm(
                    v, vector_size);

            // Compute distance between the j-th vector and vector v
            distance = sqrt(norm_v*norm_v - 2.0*inner_prod + norm*norm);

            // If distance is zero, do not reorthogonalize i-th against
            // the j-th vector.
            if (distance < 2.0 * epsilon * sqrt(vector_size))
            {
                continue;
            }
        }

        // Subtraction
        cVectorOperations<DataType>::subtract_scaled_vector(
                &V[vector_size*i], vector_size, scale, v);
    }
}


// =====================
// orthogonalize vectors
// =====================

/// \brief         Orthogonalizes set of vectors mutually using modified
///                Gram-Schmidt process.
///
/// \note          Let \c m be the number of vectors (\c num_vectors), and
///                let \c n be the size of each vector (\c vector_size). In
///                general, \c n is much larger (large matrix size), and \c m
///                is small, in order of a couple of hundred. But for small
///                matrices (where \c n could be smaller then \c m), then each
///                vector can be orthogonalized at most to \c n other vectors.
///                This is because the dimension of the vector space is \c n.
///                Thus, if there are extra vectors, each vector is
///                orthogonalized to window of the previous \c n vector.
///
///                If one of the column vectors is identical to one of other
///                column vectors in \c V, one of the vectors is regenerated
///                by random array and the orthogonalization is repeated.
///
/// \note          If two vectors are identical (or the norm of their
///                difference is very small), they cannot be orthogonalized
///                against each other. In this case, one of the vectors is
///                re-generated by new random numbers.
///                
/// \warning       if \c num_vectors is larger than \c vector_size, the
///                orthogonalization fails since not all vectors are
///                independent, and at least one vector becomes zero.
///
/// \param[in,out] vectors
///                2D array of size \c vector_size*num_vectors. This array will
///                be modified in-place and will be output of this function.
///                Note that this is Fortran ordering, meaning that the first
///                index is contiguous. Hence, to call the j-th element of the
///                i-th vector, use \c &vectors[i*vector_size + j].
/// \param[in]     num_vectors
///                Number of columns of vectors array.
/// \param[in]     vector_size
///                Number of rows of vectors array.

static void orthogonalize_vectors(
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors)
{
    // Do nothing if there is only one vector
    if (num_vectors < 2)
    {
        return;
    }

    IndexType i = 0;
    IndexType j;
    IndexType start = 0;
    DataType inner_prod;
    DataType norm;
    DataType norm_i;
    DataType distance;
    DataType epsilon = std::numeric_limits<DataType>::epsilon();
    IndexType success = 1;
    IndexType max_num_trials = 20;
    IndexType num_trials = 0;
    IndexType num_threads = 1;
    auto random_number_generator = ThreadedRNG64(num_threads);

    while (i < num_vectors)
    {
        if ((success == 0) && (num_trials >= max_num_trials))
        {
            std::cerr << "ERROR: Cannot orthogonalize vectors after " \
                      << num_trials << " trials. Aborting." \
                      << std::endl;
            abort();
        }

        // Reset on new trial (if it was set to 0 before to start a new trial)
        success = 1;

        // j iterates on previous vectors in a window of at most vector_size
        if (static_cast<LongIndexType>(i) > vector_size)
        {
            // When vector_size is smaller than i, it is fine to cast to signed
            start = i - static_cast<IndexType>(vector_size);
        }

        // Reorthogonalize against previous vectors
        for (j=start; j < i; ++j)
        {
            // Norm of the j-th vector
            norm = cVectorOperations<DataType>::euclidean_norm(
                    &vectors[j*vector_size], vector_size);

            // Check norm
            if (norm < epsilon * sqrt(vector_size))
            {
                std::cerr << "WARNING: norm of the given vector is too " \
                          << " small. Cannot reorthogonalize against zero" \
                          << "vector. Skipping."
                          << std::endl;
                continue;
            }

            // Projecting i-th vector to j-th vector
            inner_prod = cVectorOperations<DataType>::inner_product(
                    &vectors[i*vector_size], &vectors[j*vector_size],
                    vector_size);

            // Scale of subtraction
            DataType scale = inner_prod / (norm * norm);

            // If scale is is 1, it is possible that i-th and j-th vectors are
            // identical (or close). So, instead of subtracting them,
            // regenerate new i-th vector.
            if (std::abs(scale - 1.0) <= 2.0 * epsilon)
            {
                // Norm of the i-th vector
                norm_i = cVectorOperations<DataType>::euclidean_norm(
                        &vectors[i*vector_size], vector_size);

                // Compute distance between i-th and j-th vector
                distance = sqrt(norm_i*norm_i - 2.0*inner_prod + norm*norm);

                // If distance is zero, do not reorthogonalize i-th against
                // vector j-th and the subsequent vectors after j-th.
                if (distance < 2.0 * epsilon * sqrt(vector_size))
                {
                    // Regenerate new random vector for i-th vector
                    VectorGenerator<DataType>::generate_random_array(
                            random_number_generator, &vectors[i*vector_size],
                            vector_size, num_threads
                    );

                    // Repeat the reorthogonalization for i-th vector against
                    // all previous vectors again.
                    success = 0;
                    ++num_trials;
                    break;
                }
            }

            // Subtraction
            cVectorOperations<DataType>::subtract_scaled_vector(
                    &vectors[vector_size*j], vector_size, scale,
                    &vectors[vector_size*i]);

            // Norm of the i-th vector
            norm_i = cVectorOperations<DataType>::euclidean_norm(
                    &vectors[i*vector_size], vector_size);

            // If the norm is too small, regenerate the i-th vector randomly
            if (norm_i < epsilon * sqrt(vector_size))
            {
                // Regenerate new random vector for i-th vector
                VectorGenerator<DataType>::generate_random_array(
                        random_number_generator, &vectors[i*vector_size],
                        vector_size, num_threads
                );

                // Repeat the reorthogonalization for i-th vector against
                // all previous vectors again.
                success = 0;
                ++num_trials;
                break;
            }
        }

        if (success == 1)
        {
            ++i;

            // Reset if num_trials was incremented before.
            num_trials = 0;
        }
    }
}

};

// ===============================
// Explicit template instantiation
// ===============================

template struct cOrthogonalization<float>;
template struct cOrthogonalization<double>;
template struct cOrthogonalization<long double>;

#endif  // _ORTHOGONALIZATION_H_


