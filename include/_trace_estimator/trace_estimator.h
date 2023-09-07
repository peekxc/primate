/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _TRACE_ESTIMATOR_TRACE_ESTIMATOR_H_
#define _TRACE_ESTIMATOR_TRACE_ESTIMATOR_H_

#include <omp.h>  // omp_set_num_threads
#include <cmath>  // sqrt, pow
#include <cstddef>  // NULL
#include <concepts>     // invocable 
#include "convergence_tools.h"  // check_convergence, average_estimates
#include "../_timer/timer.h"  // Timer
#include "../_random_generator/random_concepts.h"  // ThreadSafeRBG
#include "../_random_generator/vector_generator.h"  // vector generator
#include "../_diagonalization/diagonalization.h"  // Diagonalization
#include "../_diagonalization/lanczos_tridiagonalization.h"  // c_lanczos_tridiagonalization
#include "../_diagonalization/golub_kahn_bidiagonalization.h"  // c_golub_kahn_bidiagonaliza...

// template < typename DataType >
// struct cTraceEstimator {

    // =================
    // c trace estimator
    // =================

    /// \brief      Stochastic Lanczos quadrature method to estimate trace of a
    ///             function of a linear operator. Both function and the linear
    ///             operator can be defined with parameters.
    ///
    /// \details    Multiple batches of parameters of the linear operator can be
    ///             passed to this function. In such a case, the output trace is an
    ///             array of the of the number of the inquired parameters.
    ///
    ///             The stochastic estimator computes multiple samples of trace and
    ///             the final result is the average of the samples. This function
    ///             outputs both the samples of estimated trace values (in
    ///             \c samples array) and their average (in \c trace array).
    ///
    /// \param[in]  A
    ///             An instance of a class derived from \c LinearOperator class.
    ///             This object will perform the matrix-vector operation and/or
    ///             transposed matrix-vector operation for a linear operator. The
    ///             linear operator can represent a fixed matrix, or a combination
    ///             of matrices together with some given parameters.
    /// \param[in]  parameters
    ///             The parameters of the linear operator \c A. The size of this
    ///             array is \c num_parameters*num_inquiries where
    ///             \c num_parameters is the number of parameters that define the
    ///             linear operator \c A, and \c num_inquiries is the number of
    ///             different set of parameters to compute trace on different
    ///             parametrized operators. The j-th set of parameters are stored
    ///             in \c parameters[j*num_parameters:(j+1)*num_parameters]. That
    ///             is, this array is contiguous over each batch of parameters.
    /// \param[in]  num_inquiries
    ///             The number of batches of parameters. This function computes
    ///             \c num_inquiries values of trace corresponding to different
    ///             batch of parameters of the linear operator \c A. Hence, the
    ///             number of output trace is \c num_inquiries. Hence, it is the
    ///             number of columns of the output array \c samples.
    /// \param[in]  matrix_function
    ///             An instance of \c Function class which has the function
    ///             \c function. This function defines the matrix function, and
    ///             operates on scalar eigenvalues of the matrix.
    /// \param[in]  gram
    ///             Flag indicating whether the linear operator \c A is Gramian.
    ///             If the linear operator is:
    ///             * Gramian, then, Lanczos tridiagonalization method is
    ///               employed. This method requires only matrix-vector
    ///               multiplication.
    ///             * not Gramian, then, Golub-Kahn bidiagonalization method is
    ///               employed. This method requires both matrix and
    ///               transposed-matrix vector multiplications.
    /// \param[in]  exponent
    ///             The exponent parameter \c p in the trace of the expression
    ///             \f$ f((\mathbf{A} + t \mathbf{B})^p) \f$. The exponent is a
    ///             real number and by default it is set to \c 1.0.
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
    /// \param[in]  lanczos_degree
    ///             The number of Lanczos recursive iterations. The operator \c A
    ///             is reduced to a square tridiagonal (or bidiagonal) matrix of
    ///             the size \c lanczos_degree. The eigenvalues (or singular
    ///             values) of this reduced matrix is computed and used in the
    ///             stochastic Lanczos quadrature method. The larger Lanczos degree
    ///             leads to a better estimation. The computational cost is
    ///             quadratically increases with the lanczos degree.
    /// \param[in]  lanczos_tol
    ///             The tolerance to stop the Lanczos recursive iterations before
    ///             the end of iterations reached. If the tolerance is not met, the
    ///             iterations (total of \c lanczos_degree iterations) continue
    ///             till end.
    /// \param[in]  min_num_samples
    ///             Minimum number of times that the trace estimation is repeated.
    ///             Within the min number of samples, the Monte-Carlo continues
    ///             even if convergence is reached.
    /// \param[in]  max_num_samples
    ///             The number of times that the trace estimation is repeated. The
    ///             output trace value is the average of the samples. Hence, this
    ///             is the number of rows of the output array \c samples. Larger
    ///             number of samples leads to a better trace estimation. The
    ///             computational const linearly increases with number of samples.
    /// \param[in]  error_atol
    ///             Absolute tolerance criterion for early termination during the
    ///             computation of trace samples. If the tolerance is not met, then
    ///             all iterations (total of \c max_num_samples) proceed till end.
    /// \param[in]  error_rtol
    ///             Relative tolerance criterion for early termination during the
    ///             computation of trace samples. If the tolerance is not met,
    ///             then all iterations (total of \c max_num_samples) proceed till
    ///             end.
    /// \param[in]  confidence_level
    ///             The confidence level of the error, which is a number between
    ///             \c 0 and \c 1. This affects the scale of \c error.
    /// \param[in]  outlier_significance_level
    ///             One minus the confidence level of the uncertainty band of the
    ///             outlier. This is a number between \c 0 and \c 1. Confidence
    ///             level of outleir and significance level of outlier are
    ///             complement of each other.
    /// \param[in]  num_threads
    ///             Number of OpenMP parallel processes. The parallelization is
    ///             implemented over the Monte-Carlo iterations.
    /// \param[out] trace
    ///             The output trace of size \c num_inquiries. These values are the
    ///             average of the rows of \c samples array.
    /// \param[out] error
    ///             The error of estimation of trace, which is the standard
    ///             deviation of the rows of \c samples array. The size of this
    ///             array is \c num_inquiries.
    /// \param[out] samples
    ///             2D array of all estimated trace samples. The shape of this
    ///             array is \c (max_num_samples*num_inquiries). The average of the
    ///             rows is also given in \c trace array.
    /// \param[out] processed_samples_indices
    ///             A 1D array indicating the processing order of rows of the
    ///             \c samples. In parallel processing, this order of processing
    ///             the rows of \c samples is not necessarly sequential.
    /// \param[out] num_samples_used
    ///             1D array of the size of the number of columns of \c samples.
    ///             Each element indicates how many iterations were used till
    ///             convergence is reached for each column of the \c samples. The
    ///             number of iterations should be a number between
    ///             \c min_num_samples and \c max_num_samples.
    /// \param[out] num_outliers
    ///             1D array with the size of number of columns of \c samples. Each
    ///             element indicates how many rows of the \c samples array were
    ///             outliers and were removed during averaging rows of \c samples.
    /// \param[out] converged
    ///             1D array of the size of the number of columns of \c samples.
    ///             Each element indicates which column of \c samples has converged
    ///             to the tolerance criteria. Normally, if the \c num_samples used
    ///             is less than \c max_num_samples, it indicates that the
    ///             convergence has reached.
    /// \param[out] alg_wall_time
    ///             The elapsed time that takes for the SLQ algorithm. This does
    ///             not include array allocation/deallocation.
    /// \return     A signal to indicate the status of computation:
    ///             * \c 1 indicates successful convergence within the given
    ///               tolerances was met. Convergence is achieved when all elements
    ///               of \c convergence array are below \c convergence_atol or
    ///               \c convergence_rtol times \c trace.
    ///             * \c 0 indicates the convergence criterion was not met for at
    ///               least one of the trace inquiries.
    template< bool gramian, std::floating_point DataType, Operator Matrix, std::invocable< DataType > Func > 
    FlagType trace_estimator(
            Matrix* A,
            DataType* parameters,
            const IndexType num_inquiries,
            Func&& matrix_function,
            const DataType exponent,
            const FlagType orthogonalize,
            const IndexType lanczos_degree,
            const DataType lanczos_tol,
            const IndexType min_num_samples,
            const IndexType max_num_samples,
            const DataType error_atol,
            const DataType error_rtol,
            const DataType confidence_level,
            const DataType outlier_significance_level,
            const IndexType num_threads,
            DataType* trace,
            DataType* error,
            DataType** samples,
            IndexType* processed_samples_indices,
            IndexType* num_samples_used,
            IndexType* num_outliers,
            FlagType* converged,
            float& alg_wall_time)
    {
        IndexType matrix_size = static_cast< std::pair< size_t, size_t > >(A->shape()).first;

        // Set the number of threads
        omp_set_num_threads(num_threads);

        // Allocate 1D array of random vectors We only allocate a random vector
        // per parallel thread. Thus, the total size of the random vectors is
        // matrix_size*num_threads. On each iteration in parallel threads, the
        // alocated memory is resued. That is, in each iteration, a new random
        // vector is generated for that specific thread id.
        IndexType random_vectors_size = matrix_size * num_threads;
        DataType* random_vectors = new DataType[random_vectors_size];

        // Thread-safe random bit generator
        auto rng = ThreadedRNG64(num_threads);

        // The counter of filled size of processed_samples_indices array
        // This scalar variable is defined as array to be shared among al threads
        IndexType num_processed_samples = 0;

        // Criterion for early termination of iterations if convergence reached
        // This scalar variable is defined as array to be shared among al threads
        FlagType all_converged = 0;

        // Using square-root of max possible chunk size for parallel schedules
        unsigned int chunk_size = static_cast<int>(
            sqrt(static_cast<DataType>(max_num_samples) / num_threads)
        );
        if (chunk_size < 1) {
            chunk_size = 1;
        }

        // Timing elapsed time of algorithm
        Timer timer;
        timer.start();

        // Shared-memory parallelism over Monte-Carlo ensemble sampling
        IndexType i;
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (i=0; i < max_num_samples; ++i) {
            if (!static_cast<bool>(all_converged)) {
                int thread_id = omp_get_thread_num();

                // Perform one Monte-Carlo sampling to estimate trace
                stochastic_lanczos_quadrature< gramian >(
                    A, parameters, num_inquiries, matrix_function,
                    exponent, orthogonalize, lanczos_degree, lanczos_tol,
                    rng,
                    &random_vectors[matrix_size*thread_id], converged,
                    samples[i]
                );

                #pragma omp critical
                {
                    // Store the index of processed samples
                    processed_samples_indices[num_processed_samples] = i;
                    ++num_processed_samples;

                    // Check whether convergence criterion has been met to stop.
                    // This check can also be done after another parallel thread
                    // set all_converged to "1", but we continue to update error.
                    all_converged = ConvergenceTools< DataType >::check_convergence(
                        samples, min_num_samples, num_inquiries,
                        processed_samples_indices, num_processed_samples,
                        confidence_level, error_atol, error_rtol, error,
                        num_samples_used, converged
                    );
                } // end critical section
            }
        }

        // Elapsed wall time of the algorithm (computation only, not array i/o)
        timer.stop();
        alg_wall_time = timer.elapsed();

        // Remove outliers from trace estimates and average trace estimates
        ConvergenceTools<DataType>::average_estimates(
            confidence_level, outlier_significance_level, num_inquiries,
            max_num_samples, num_samples_used, processed_samples_indices,
            samples, num_outliers, trace, error
        );

        // Deallocate memory
        delete[] random_vectors;

        return all_converged;
    }


    // ===============================
    // c stochastic lanczos quadrature
    // ===============================

    /// \brief      For a given random input vector, computes one Monte-Carlo
    ///             sample to estimate trace using Lanczos quadrature method.
    ///
    /// \note       In special case when an eigenvalue relation is known, this
    ///             function sets the converged inquiries to "not" converged in
    ///             order to continue updating those inquiries. This is because in
    ///             this special case, computing for other inquiries is free.
    ///
    /// \param[in]  A
    ///             An instance of a class derived from \c LinearOperator class.
    ///             This object will perform the matrix-vector operation and/or
    ///             transposed matrix-vector operation for a linear operator. The
    ///             linear operator can represent a fixed matrix, or a combination
    ///             of matrices together with some given parameters.
    /// \param[in]  parameters
    ///             The parameters of the linear operator \c A. The size of this
    ///             array is \c num_parameters*num_inquiries where
    ///             \c num_parameters is the number of parameters that define the
    ///             linear operator \c A, and \c num_inquiries is the number of
    ///             different set of parameters to compute trace on different
    ///             parametrized operators. The j-th set of parameters are stored
    ///             in \c parameters[j*num_parameters:(j+1)*num_parameters]. That
    ///             is, this array is contiguous over each batch of parameters.
    /// \param[in]  num_inquiries
    ///             The number of batches of parameters. This function computes
    ///             \c num_inquiries values of trace corresponding to different
    ///             batch of parameters of the linear operator \c A. Hence, the
    ///             number of output trace is \c num_inquiries. Hence, it is the
    ///             number of columns of the output array \c samples.
    /// \param[in]  matrix_function
    ///             An instance of \c Function class which has the function \c
    ///             function. This function defines the matrix function, and
    ///             operates on scalar eigenvalues of the matrix.
    /// \param[in]  gram
    ///             Flag indicating whether the linear operator \c A is Gramian.
    ///             If the linear operator is:
    ///             * Gramian, then, Lanczos tridiagonalization method is
    ///               employed. This method requires only matrix-vector
    ///               multiplication.
    ///             * not Gramian, then, Golub-Kahn bidiagonalization method is
    ///               employed. This method requires both matrix and
    ///               transposed-matrix vector multiplications.
    /// \param[in]  exponent
    ///             The exponent parameter \c p in the trace of the expression
    ///             \f$ f((\mathbf{A} + t \mathbf{B})^p) \f$. The exponent is a
    ///             real number and by default it is set to \c 1.0.
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
    /// \param[in]  lanczos_degree
    ///             The number of Lanczos recursive iterations. The operator \c A
    ///             is reduced to a square tridiagonal (or bidiagonal) matrix of
    ///             the size \c lanczos_degree. The eigenvalues (or singular
    ///             values) of this reduced matrix is computed and used in the
    ///             stochastic Lanczos quadrature method. The larger Lanczos degre
    ///             leads to a better estimation. The computational cost is
    ///             quadratically increases with the lanczos degree.
    /// \param[in]  lanczos_tol
    ///             The tolerance to stop the Lanczos recursive iterations before
    ///             the end of iterations reached. If the tolerance is not met, the
    ///             iterations (total of \c lanczos_degree iterations) continue
    ///             till end.
    /// \param[in]  random_number_generator
    ///             Generates random numbers that fills \c random_vector. In each
    ///             parallel thread, an independent sequence of random numbers are
    ///             generated. This object should be initialized by \c num_threads.
    /// \param[in]  random_vector
    ///             A 1D vector of the size of matrix \c A. The Lanczos iterations
    ///             start off from this random vector. Each given random vector is
    ///             used per a Monte-Carlo computation of the SLQ method. In the
    ///             Lanczos iterations, other vectors are generated orthogonal to
    ///             this initial random vector. This array is filled inside this
    ///             function.
    /// \param[out] converged
    ///             1D array of the size of the number of columns of \c samples.
    ///             Each element indicates which column of \c samples has converged
    ///             to the tolerance criteria. Normally, if the \c num_samples used
    ///             is less than \c max_num_samples, it indicates that the
    ///             convergence has reached.
    /// \param[out] trace_estimate
    ///             1D array of the size of the number of columns of \c samples
    ///             array. This array constitures each row of \c samples array.
    ///             Each element of \c trace_estimates is the estimated trace for
    ///             each parameter inquiry.

    template < bool gramian, std::floating_point DataType, AffineOperator Matrix, std::invocable< DataType > Func, ThreadSafeRBG RBG >
    void stochastic_lanczos_quadrature(
        Matrix* A,
        DataType* parameters,
        const IndexType num_inquiries,
        Func&& matrix_function,
        const DataType exponent,
        const FlagType orthogonalize,
        const IndexType lanczos_degree,
        const DataType lanczos_tol,
        RBG& rng,
        DataType* random_vector,
        FlagType* converged,
        DataType* trace_estimate)
    {
        // Matrix size
        IndexType matrix_size = static_cast< std::pair< size_t, size_t > >(A->shape()).first;

        // Fill random vectors with Rademacher distribution (+1, -1), normalized
        // but not orthogonalized. Settng num_threads to zero indicates to not
        // create any new threads in RandomNumbrGenerator since the current
        // function is inside a parallel thread.
        IndexType num_threads = 0;
        VectorGenerator< DataType >::generate_array(
            rng, random_vector, matrix_size, num_threads
        ); 

        // Allocate diagonals (alpha) and supdiagonals (beta) of Lanczos matrix
        DataType* alpha = new DataType[lanczos_degree];
        DataType* beta = new DataType[lanczos_degree];

        // Define 2D arrays needed to decomposition. All these arrays are
        // defined as 1D array with Fortran ordering
        DataType* eigenvectors = NULL;
        DataType* left_singularvectors = NULL;
        DataType* right_singularvectors_transposed = NULL;

        // Actual number of inquiries
        IndexType required_num_inquiries = num_inquiries;
        // if (A->is_eigenvalue_relation_known())
        // {
        //     // When a relation between eigenvalues and the parameters of the linear
        //     // operator is known, to compute eigenvalues of for each inquiry, only
        //     // computing one inquiry is enough. This is because an eigenvalue for
        //     // one parameter setting is enough to compute eigenvalue of another set
        //     // of parameters.
        //     required_num_inquiries = 1;
        // }

        // Allocate and initialize theta
        IndexType i;
        IndexType j;
        DataType** theta = new DataType*[num_inquiries];
        for (j=0; j < num_inquiries; ++j) {
            theta[j] = new DataType[lanczos_degree];
            for (i=0; i < lanczos_degree; ++i) {
                theta[j][i] = 0.0;  // Initialize components to zero
            }
        }

        // Allocate and initialize tau
        DataType** tau = new DataType*[num_inquiries];
        for (j=0; j < num_inquiries; ++j) {
            tau[j] = new DataType[lanczos_degree];
            for (i=0; i < lanczos_degree; ++i) {
                tau[j][i] = 0.0;   // Initialize components to zero
            }
        }

        // Allocate lanczos size for each inquiry. This variable keeps the non-zero
        // size of the tri-diagonal (or bi-diagonal) matrix. Ideally, this matrix
        // is of the size lanczos_degree. But, due to the early termination, this
        // size might be smaller.
        IndexType* lanczos_size = new IndexType[num_inquiries];

        // Number of parameters of linear operator A
        IndexType num_parameters = A->get_num_parameters();
        // IndexType num_parameters = A->parameters.size();

        // Lanczos iterations, computes theta and tau for each inquiry parameter
        for (j=0; j < required_num_inquiries; ++j) {
            // If trace is already converged, do not compute on the new sample.
            // However, exclude the case where required_num_inquiries is not the
            // same as num_inquiries, since in this case, we compute one inquiry
            // for multiple parameters.
            if ((converged[j] == 1) && (required_num_inquiries == num_inquiries)) {
                continue; // MJP: why isn't this a break?
            }

            // Set parameter of linear operator A
            A->set_parameters(&parameters[j*num_parameters]);
            if constexpr (gramian) {
                // Use Golub-Kahn-Lanczos Bi-diagonalization
                lanczos_size[j] = golub_kahn_bidiagonalization(
                    A, random_vector, matrix_size, lanczos_degree, lanczos_tol,
                    orthogonalize, alpha, beta
                );

                // Allocate matrix of singular vectors (1D array, Fortran ordering)
                left_singularvectors = new DataType[lanczos_size[j] * lanczos_size[j]];
                right_singularvectors_transposed = new DataType[lanczos_size[j] * lanczos_size[j]];

                // Note: alpha is written in-place with singular values
                svd_bidiagonal< DataType >(
                    alpha, beta, left_singularvectors,
                    right_singularvectors_transposed, lanczos_size[j]
                );

                // theta and tau from singular values and vectors
                for (i=0; i < lanczos_size[j]; ++i) {
                    theta[j][i] = alpha[i] * alpha[i];
                    tau[j][i] = right_singularvectors_transposed[i];
                }
            } else {
                // Use Lanczos Tri-diagonalization
                lanczos_size[j] = lanczos_tridiagonalization(
                    A, random_vector, matrix_size, lanczos_degree, lanczos_tol,
                    orthogonalize, alpha, beta
                );

                // Allocate eigenvectors matrix (1D array with Fortran ordering)
                // MJP: why is this allocated inside here?? why not re-use memory? 
                // Could just allocate memory of size (lanczos_degree * lanczos_degree) which is thread-specific
                // The lanczos_size[j] should never exceed lanczos_degree, and given to eigh_tridiagonal should be safe
                eigenvectors = new DataType[lanczos_size[j] * lanczos_size[j]];

                // Note: alpha is written in-place with eigenvalues
                eigh_tridiagonal< DataType >(
                    alpha, beta, eigenvectors, lanczos_size[j]
                );

                // theta and tau from singular values and vectors
                for (i=0; i < lanczos_size[j]; ++i) {
                    theta[j][i] = alpha[i];
                    tau[j][i] = eigenvectors[i * lanczos_size[j]];
                }
            }
        }

        // If an eigenvalue relation is known, compute the rest of eigenvalues
        // using the eigenvalue relation given in the operator A for its
        // eigenvalues. If no eigenvalue relation is not known, the rest of
        // eigenvalues were already computed in the above loop and no other
        // computation is needed.
        // if (A->is_eigenvalue_relation_known() && num_inquiries > 1) {
        //     // When the code execution reaches this function, at least one of the
        //     // inquiries is not converged, but some others might have been
        //     // converged already. Here, we force-update those that are even
        //     // converged already by setting converged to false. The extra update is
        //     // free of charge when a relation for the eigenvalues are known.
        //     for (j=0; j < num_inquiries; ++j)
        //     {
        //         converged[j] = 0;
        //     }

        //     // Compute theta and tau for the rest of inquiry parameters
        //     for (j=1; j < num_inquiries; ++j)
        //     {
        //         // Only j=0 was iterated before. Set the same size for other j-s
        //         lanczos_size[j] = lanczos_size[0];

        //         for (i=0; i < lanczos_size[j]; ++i)
        //         {
        //             // Shift eigenvalues by the old and new parameters
        //             theta[j][i] = A->get_eigenvalue(
        //                     &parameters[0],
        //                     theta[0][i],
        //                     &parameters[j*num_parameters]);

        //             // tau is the same (at least for the affine operator)
        //             tau[j][i] = tau[0][i];
        //         }
        //     }
        // }

        // Estimate trace using quadrature method
        DataType quadrature_sum;
        for (j=0; j < num_inquiries; ++j) {
            // If the j-th inquiry is already converged, skip.
            if (converged[j] == 1) {
                continue;
            }

            // Initialize sum for the integral of quadrature
            quadrature_sum = 0.0;

            // Important: This loop should iterate till lanczos_size[j], but not
            // lanczos_degree. Otherwise the computation is wrong for certain
            // matrices, such as if the input matrix is identity, or rank
            // deficient. By using lanczos_size[j] instead of lanczos_degree, all
            // issues with special matrices will resolve.
            for (i=0; i < lanczos_size[j]; ++i) {
                quadrature_sum += tau[j][i] * tau[j][i] * matrix_function(pow(theta[j][i], exponent));
            }

            trace_estimate[j] = matrix_size * quadrature_sum;
        }

        // Release dynamic memory
        delete[] alpha;
        delete[] beta;
        delete[] lanczos_size;

        for (j=0; j < required_num_inquiries; ++j) {
            delete[] theta[j];
        }
        delete[] theta;

        for (j=0; j < required_num_inquiries; ++j) {
            delete[] tau[j];
        }
        delete[] tau;

        if (eigenvectors != NULL) {
            delete[] eigenvectors;
        }

        if (left_singularvectors != NULL) {
            delete[] left_singularvectors;
        }

        if (right_singularvectors_transposed != NULL) {
            delete[] right_singularvectors_transposed;
        }
    }
// };

// ===============================
// Explicit template instantiation
// ===============================

// template struct cTraceEstimator<float>;
// template struct cTraceEstimator<double>;
// template struct cTraceEstimator<long double>;

#endif