/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _GOLUB_KAHN_BIDIAGONALIZATION_H_
#define _GOLUB_KAHN_BIDIAGONALIZATION_H_

// =======
// Headers
// =======

#include <cmath>  // sqrt
#include "../_definitions/types.h"  // IndexType, LongIndexType, FlagType
#include "../_linear_operator/linear_operator.h"  // cLinearOperator
#include "../_orthogonalization/orthogonalization.h"  // cOrthogonalization
#include "../_c_basic_algebra/c_vector_operations.h"  // cVectorOperations


// ============================
// golub-kahn bidiagonalization
// ============================

/// \brief      Bi-diagonalizes the positive-definite matrix \c A using
///             Golub-Kahn-Lanczos method.
///
/// \details    This method bi-diagonalizes matrix \c A to \c B using the start
///             vector \c w. \c m is the Lanczos degree, which will be the size
///             of square matrix \c B.
///
///             The output of this function are \c alpha (of length \c m) and
///             \c beta (of length \c m+1) which are diagonal (\c alpha[:]) and
///             off-diagonal (\c beta[1:]) elements of the bi-diagonal \c (m,m)
///             symmetric and positive-definite matrix \c B.
///
///             #### Lanczos tridiagonalization vs Golub-Kahn Bidiagonalization
///             * The Lanczos tri-diagonalization is twice faster (in runtime),
///               as it has only one matrix-vector multiplication. Whereas the
///               Golub-Kahn bi-diagonalization has two matrix-vector
///               multiplications.
///             * The Lanczos tri-diagonalization can only be applied to
///               symmetric matrices. Whereas the Golub-Kahn bi-diagonalization
///               can be applied to any matrix.
///
///             #### Reference
///
///             * NetLib Algorithm 6.27,
///               netlib.org/utk/people/JackDongarra/etemplates/node198.html
///             * Matrix Computations, Golub, p. 495
///             * Demmel, J., Templates for Solution of Algebraic Eigenvalue
///               Problem, p. 143
///
/// \warning    When the matrix \c A is very close to the identity matrix, the
///             Golub-Kahn bi-diagonalization method can not find \c beta, as
///             \c beta becomes zero. If \c A is not exactly identity, you may
///             decrease the Tolerance to a very small number. However, if \c A
///             is almost identity matrix, decreasing \c lanczos_tol will not
///             help, and this function cannot be used.
///
/// \sa         lanczos_tridiagonalizaton
///
/// \param[in]  A
///             A linear operator that represents a matrix of size \c (n,n) and
///             can perform matrix-vector operation with \c dot() method and
///             transposed matrix-vector operation with \c transpose_dot()
///             method. This matrix should be positive-definite.
/// \param[in]  v
///             Start vector for the Lanczos tri-diagonalization. Column vector
///             of size \c n. It could be generated randomly. Often it is
///             generated by the Rademacher distribution with entries \c +1 and
///             \c -1.
/// \param[in]  n
///             Size of the square matrix \c A, which is also the size of the
///             vector \c v.
/// \param[in]  m
///             Lanczos degree, which is the number of Lanczos iterations.
/// \param[in]  lanczos_tol
///             The tolerance of the residual error of the Lanczos iteration.
/// \param[in]  orthogonalize
///             Indicates whether to orthogonalize the orthogonal eigenvectors
///             during Lanczos recursive iterations.
///             * If set to \c 0, no orthogonalization is performed.
///             * If set to a negative integer, a newly computed eigenvector is
///               orthogonalized against all the previous eigenvectors (full
///               reorthogonalization).
///             * If set to a positive integer, say \c q less than
///               \c lanczos_degree, the newly computed eigenvector is
///               orthogonalized against the last \c q previous eigenvectors
///               (partial reorthogonalization).
///             * If set to an integer larger than \c lanczos_degree, it is cut
///               to \c lanczos_degree, which effectively orthogonalizes
///               against all previous eigenvectors (full reorthogonalization).
/// \param[out] alpha
///             This is a 1D array of size \c m and \c alpha[:] constitute the
///             diagonal elements of the bi-diagonal matrix \c B. This is the
///             output and written in place.
/// \param[out] beta
///             This is a 1D array of size \c m, and the elements \c beta[:]
///             constitute the sup-diagonals of the bi-diagonal matrix \c B.
///             This array is the output and written in place.
/// \return     Counter for the Lanczos iterations. Normally, the size of the
///             output matrix should be \c (m,m), which is the Lanczos degree.
///             However, if the algorithm terminates early, the size of \c
///             alpha and \c beta, and hence the output tri-diagonal matrix, is
///             smaller. This counter keeps track of the *non-zero* size of \c
///             alpha and \c beta.

template< std::floating_point DataType, AdjointOperator Matrix >
IndexType golub_kahn_bidiagonalization(
        Matrix* A,
        const DataType* v,
        const LongIndexType n,
        const IndexType m,
        const DataType lanczos_tol,
        const FlagType orthogonalize,
        DataType* alpha,
        DataType* beta)
{
    // buffer_size is number of last orthogonal vectors to keep in buffers U, V
    IndexType buffer_size;
    if (orthogonalize == 0) {
        // At least two vectors must be stored in buffer for Lanczos recursion
        buffer_size = 2;
    }
    else if ((orthogonalize < 0) || (orthogonalize > static_cast<FlagType>(m) - 1)) {
        // Using full reorthogonalization, keep all of the m vectors in buffer
        buffer_size = m;
    }
    else {
        // Orthogonalize with less than m vectors (0 < orthogonalize < m-1)
        // plus one vector for the latest (the j-th) vector
        buffer_size = orthogonalize + 1;
    }

    // Allocate 2D array (as 1D array, and coalesced row-wise) to store
    // the last buffer_size of orthogonalized vectors of length n. New vectors
    // are stored by cycling through the buffer to replace with old ones.
    DataType* U = new DataType[n * buffer_size];
    DataType* V = new DataType[n * buffer_size];

    // Normalize vector v and copy to v_old
    cVectorOperations<DataType>::normalize_vector_and_copy(v, n, &V[0]);

    // Declare iterators
    IndexType j;
    IndexType lanczos_size = 0;
    IndexType num_ortho;

    // Golub-Kahn iteration
    for (j=0; j < m; ++j) {
        // Counter for the non-zero size of alpha and beta
        ++lanczos_size;

        // u_new = A.dot(v_old)
        A->matvec(&V[(j % buffer_size)*n], &U[(j % buffer_size)*n]);

        // Performing: u_new[i] = u_new[i] - beta[j] * u_old[i]
        if (j > 0){
            cVectorOperations<DataType>::subtract_scaled_vector(
                    &U[((j-1) % buffer_size)*n], n, beta[j-1],
                    &U[(j % buffer_size)*n]);
        }

        // orthogonalize u_new against previous vectors
        if (orthogonalize != 0) {
            // Find how many column vectors are filled so far in the buffer V
            num_ortho = j < buffer_size ? j : buffer_size - 1;

            // Gram-Schmidt process
            if (j > 0) {
                cOrthogonalization<DataType>::gram_schmidt_process(
                        &U[0], n, buffer_size, (j-1)%buffer_size, num_ortho,
                        &U[(j % buffer_size)*n]);
            }
        }

        // Normalize u_new and set its norm to alpha[j]
        alpha[j] = cVectorOperations<DataType>::normalize_vector_in_place(&U[(j % buffer_size)*n], n);

        // Performing: v_new = A.T.dot(u_new) - alpha[j] * v_old
        A->rmatvec(&U[(j % buffer_size)*n], &V[((j+1) % buffer_size)*n]);

        // Performing: v_new[i] = v_new[i] - alpha[j] * v_old[i]
        cVectorOperations<DataType>::subtract_scaled_vector(
                &V[(j % buffer_size)*n], n, alpha[j],
                &V[((j+1) % buffer_size)*n]);

        // orthogonalize v_new against previous vectors
        if (orthogonalize != 0) {
            cOrthogonalization<DataType>::gram_schmidt_process(
                    &V[0], n, buffer_size, j%buffer_size, num_ortho,
                    &V[((j+1) % buffer_size)*n]);
        }

        // Update beta as the norm of v_new
        beta[j] = cVectorOperations<DataType>::normalize_vector_in_place(&V[((j+1) % buffer_size)*n], n);

        // Exit criterion when the vector r is zero. If each component of a
        // zero vector has the tolerance epsilon, (which is called lanczos_tol
        // here), the tolerance of norm of r is epsilon times sqrt of n.
        if (beta[j] < lanczos_tol * sqrt(n)){
            break;
        }
    }

    // Free dynamic memory
    delete[] U;
    delete[] V;

    return lanczos_size;
}


// ===============================
// Explicit template instantiation
// ===============================

// golub kahn bidiagonalization

// template < AdjointOperator Matrix, typename DataType = Matrix::value_type >
// IndexType golub_kahn_bidiagonalization<float>(
//     AdjointOperator Matrix<float>* A,
//     const float* v,
//     const LongIndexType n,
//     const IndexType m,
//     const float lanczos_tol,
//     const FlagType orthogonalize,
//     float* alpha,
//     float* beta
// );

// template IndexType golub_kahn_bidiagonalization<double>(
//         cLinearOperator<double>* A,
//         const double* v,
//         const LongIndexType n,
//         const IndexType m,
//         const double lanczos_tol,
//         const FlagType orthogonalize,
//         double* alpha,
//         double* beta);

// template IndexType golub_kahn_bidiagonalization<long double>(
//         cLinearOperator<long double>* A,
//         const long double* v,
//         const LongIndexType n,
//         const IndexType m,
//         const long double lanczos_tol,
//         const FlagType orthogonalize,
//         long double* alpha,
//         long double* beta);


#endif 