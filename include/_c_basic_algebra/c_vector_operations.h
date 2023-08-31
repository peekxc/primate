/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_BASIC_ALGEBRA_C_VECTOR_OPERATIONS_H_
#define _C_BASIC_ALGEBRA_C_VECTOR_OPERATIONS_H_

#define USE_CBLAS 0

#include <cmath>  // sqrt
#include "./c_vector_operations.h"
#include "../_definitions/definitions.h"  // USE_CBLAS

#if (USE_CBLAS == 1)
    #include "./cblas_interface.h"
#endif

template <typename DataType>
struct cVectorOperations {

// ===========
// copy vector
// ===========

/// \brief      Copies a vector to a new vector. Result is written in-place
///
/// \param[in]  input_vector
///             A 1D array
/// \param[in]  vector_size
///             Length of vector array
/// \param[out] output_vector
///             Output vector (written in place).

static void copy_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        DataType* output_vector)
{
    #if (USE_CBLAS == 1)

    // Using Openblas
    int incx = 1;
    int incy = 1;

    cblas_interface::xcopy(vector_size, input_vector, incx, output_vector,
                           incy);

    #else

    // Not using OpenBlas
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] = input_vector[i];
    }

    #endif
}

// ==================
// copy scaled vector
// ==================

/// \brief      Scales a vector and stores to a new vector.
///
/// \param[in]  input_vector
///             A 1D array
/// \param[in]  vector_size
///             Length of vector array
/// \param[in]  scale
///             Scale coefficient to the input vector. If this is equal to one,
///             the function effectively becomes the same as \e copy_vector.
/// \param[out] output_vector
///             Output vector (written in place).

static void copy_scaled_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector)
{
    #if (USE_CBLAS == 1)

    // Using OpenBlas
    int incx = 1;
    int incy = 1;

    cblas_interface::xcopy(vector_size, input_vector, incx, output_vector,
                           incy);

    cblas_interface::xscal(vector_size, scale, output_vector, incy); 

    #else

    // Not using OpenBlas
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] = scale * input_vector[i];
    }

    #endif
}


// ======================
// subtract scaled vector
// ======================

/// \brief         Subtracts the scaled input vector from the output vector.
///
/// \details       Performs the following operation:
///                \f[
///                    \boldsymbol{b} = \boldsymbol{b} - c \boldsymbol{a},
///                \f]
///                where
///                * \f$ \boldsymbol{a} \f$ is the input vector,
///                * \f$ c \f$ is a scalar scale to the input vector, and
///                * \f$ \boldsymbol{b} \f$ is the output vector that is
///                written in-place.
///
/// \param[in]     input_vector
///                A 1D array
/// \param[in]     vector_size
///                Length of vector array
/// \param[in]     scale
///                Scale coefficient to the input vector.
/// \param[in,out] output_vector Output vector (written in place).

static void subtract_scaled_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector)
{

    #if (USE_CBLAS == 1)

    // Using OpenBlas
    int incx = 1;
    int incy = 1;

    DataType neg_scale = -scale;
    cblas_interface::xaxpy(vector_size, neg_scale, input_vector, incx,
                           output_vector, incy);

    #else

    // Not using OpenBlas
    if (scale == 0.0)
    {
        return;
    }

    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] -= scale * input_vector[i];
    }

    #endif
}


// =============
// inner product
// =============

/// \brief     Computes Euclidean inner product of two vectors.
///
/// \details   The reduction variable (here, \c inner_prod ) is of the type
///            <tt>long double</tt>. This is becase when \c DataType is \c
///            float, the summation loses the precision, especially when the
///            vector size is large. It seems that using <tt>long double</tt>
///            is slightly faster than using \c double. The advantage of using
///            a type with larger bits for the reduction variable is only
///            sensible if the compiler is optimized with \c -O2 or \c -O3
///            flags.
///
///            Using a larger bit type for the reduction variable is very
///            important for this function. If \c DataType is \c float,
///            without such consideration, the result of estimation of trace
///            can be completely wrong, just becase of the wrong inner product
///            results. For large array sizes, even libraries such as openblas
///            does not compute the dot product accurately.
///
///            The chunk computation of the dot product (as seen in the code
///            with \c chunk=5) improves the preformance with gaining twice
///            speedup. This result is not much dependet on \c chunk. For
///            example, \c chunk=10 also yields a similar result.
///
/// \param[in] vector1
///            1D array
/// \param[in] vector2
///            1D array
/// \param[in] vector_size Length of array
/// \return    Inner product of two vectors.

static DataType inner_product(
        const DataType* vector1,
        const DataType* vector2,
        const LongIndexType vector_size)
{
    #if (USE_CBLAS == 1)

    // Using OpenBlas
    int incx = 1;
    int incy = 1;

    DataType inner_prod = cblas_interface::xdot(vector_size, vector1, incx,
                                                vector2, incy);

    return inner_prod;

    #else

    // Not using OpenBlas
    long double inner_prod = 0.0;
    LongIndexType chunk = 5;
    LongIndexType vector_size_chunked = vector_size - (vector_size % chunk);

    for (LongIndexType i=0; i < vector_size_chunked; i += chunk)
    {
        inner_prod += vector1[i] * vector2[i] +
                      vector1[i+1] * vector2[i+1] +
                      vector1[i+2] * vector2[i+2] +
                      vector1[i+3] * vector2[i+3] +
                      vector1[i+4] * vector2[i+4];
    }

    for (LongIndexType i=vector_size_chunked; i < vector_size; ++i)
    {
        inner_prod += vector1[i] * vector2[i];
    }

    return static_cast<DataType>(inner_prod);

    #endif
}


// ==============
// euclidean norm
// ==============

/// \brief     Computes the Euclidean norm of a 1D array.
///
/// \details   The reduction variable (here, \c inner_prod ) is of the type
///            <tt>long double</tt>. This is becase when \c DataType is \c
///            float, the summation loses the precision, especially when the
///            vector size is large. It seems that using <tt>long double</tt>
///            is slightly faster than using \c double. The advantage of using
///            a type with larger bits for the reduction variable is only
///            sensible if the compiler is optimized with \c -O2 or \c -O3
///            flags.
///
///            Using a larger bit type for the reduction variable is very
///            important for this function. If \c DataType is \c float,
///            without such consideration, the result of estimation of trace
///            can be completely wrong, just becase of the wrong norm results.
///            For large array sizes, even libraries such as openblas does not
///            compute the dot product accurately.
///
///            The chunk computation of the dot product (as seen in the code
///            with \c chunk=5) improves the preformance with gaining twice
///            speedup. This result is not much dependet on \c chunk. For
///            example, \c chunk=10 also yields a similar result.
///
/// \param[in] vector
///            A pointer to 1D array
/// \param[in] vector_size
///            Length of the array
/// \return    Euclidean norm

static DataType euclidean_norm(
        const DataType* vector,
        const LongIndexType vector_size)
{
    #if (USE_CBLAS == 1)

    // Using OpenBlas
    int incx = 1;

    DataType norm = cblas_interface::xnrm2(vector_size, vector, incx);

    return norm;

    #else

    // Compute norm squared
    long double norm2 = 0.0;
    LongIndexType chunk = 5;
    LongIndexType vector_size_chunked = vector_size - (vector_size % chunk);

    for (LongIndexType i=0; i < vector_size_chunked; i += chunk)
    {
        norm2 += vector[i] * vector[i] +
                 vector[i+1] * vector[i+1] +
                 vector[i+2] * vector[i+2] +
                 vector[i+3] * vector[i+3] +
                 vector[i+4] * vector[i+4];
    }

    for (LongIndexType i=vector_size_chunked; i < vector_size; ++i)
    {
        norm2 += vector[i] * vector[i];
    }

    // Norm
    DataType norm = sqrt(static_cast<DataType>(norm2));

    return norm;

    #endif
}


// =========================
// normalize vector in place
// =========================

/// \brief          Normalizes a vector based on Euclidean 2-norm. The result
///                 is written in-place.
///
/// \param[in, out] vector
///                 Input vector to be normalized in-place.
/// \param[in]      vector_size
///                 Length of the input vector
/// \return         2-Norm of the input vector (before normalization)

static DataType normalize_vector_in_place(
        DataType* vector,
        const LongIndexType vector_size)
{
    #if (USE_CBLAS == 1)

    // Norm of vector
    DataType norm = euclidean_norm(
            vector, vector_size);

    // Normalize in place
    DataType scale = 1.0 / norm;
    int incx = 1;
    cblas_interface::xscal(vector_size, scale, vector, incx);

    return norm;

    #else

    // Norm of vector
    DataType norm = euclidean_norm(vector,
                                                                vector_size);

    // Normalize in place
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        vector[i] /= norm;
    }

    return norm;

    #endif
}


// =========================
// normalize vector and copy
// =========================

/// \brief      Normalizes a vector based on Euclidean 2-norm. The result is
///             written into another vector.
///
/// \param[in]  vector
///             Input vector.
/// \param[in]  vector_size
///             Length of the input vector
/// \param[out] output_vector
///             Output vector, which is the normalization of the input vector.
/// \return     2-norm of the input vector

static DataType normalize_vector_and_copy(
        const DataType* vector,
        const LongIndexType vector_size,
        DataType* output_vector)
{
    #if (USE_CBLAS == 1)

    // Norm of vector
    DataType norm = euclidean_norm(
            vector, vector_size);

    // Normalize to output
    DataType scale = 1.0 / norm;
    copy_scaled_vector(vector, vector_size, scale,
                                                    output_vector);

    return norm;

    #else

    // Norm of vector
    DataType norm = euclidean_norm(vector,
                                                                vector_size);

    // Normalize to output
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] = vector[i] / norm;
    }

    return norm;

    #endif
}

};
// ===============================
// Explicit template instantiation
// ===============================

template struct cVectorOperations<float>;
template struct cVectorOperations<double>;
template struct cVectorOperations<long double>;

#endif  // _C_BASIC_ALGEBRA_C_VECTOR_OPERATIONS_H_