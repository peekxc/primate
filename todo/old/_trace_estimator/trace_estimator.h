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
    /// \param[in]  gramian
    ///             Flag indicating whether to compute the gramian of A.
    ///             The flag semantics is as follows:
    ///             * I false, then Lanczos tridiagonalization method is
    ///               employed. This method requires only matrix-vector
    ///               multiplication.
    ///             * If true, then Golub-Kahn bidiagonalization method is
    ///               employed. This method requires both matrix and
    ///               transposed-matrix vector multiplications.
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
    template< bool gramian, std::floating_point DataType, Operator Matrix, ThreadSafeRBG RBG, std::invocable< DataType > Func > 
    FlagType trace_estimator(
        Matrix* A,
        Func&& matrix_function,
        RBG& rng,
        const IndexType distr,
        const int seed, 
        const DataType* parameters,
        const IndexType num_parameters,
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
        DataType& alg_wall_time)
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
        // std::cout << "Spinning up RNG" << std::endl;
        rng.initialize(num_threads, seed);

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

        // std::cout << "timer started: Starting SLQ" << std::endl;
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
                    A, parameters, num_parameters, matrix_function, 
                    orthogonalize, lanczos_degree, lanczos_tol,
                    rng, distr, 
                    &random_vectors[matrix_size*thread_id], converged,
                    samples[i]
                );

                #pragma omp critical
                {
                    // std::cout << "(check) sample[0] trace estimate: " << samples[i][0] << std::endl;
                    // Store the index of processed samples
                    processed_samples_indices[num_processed_samples] = i;
                    ++num_processed_samples;

                    // Check whether convergence criterion has been met to stop.
                    // This check can also be done after another parallel thread
                    // set all_converged to "1", but we continue to update error.
                    all_converged = ConvergenceTools< DataType >::check_convergence(
                        samples, min_num_samples, num_parameters,
                        processed_samples_indices, num_processed_samples,
                        confidence_level, error_atol, error_rtol, error,
                        num_samples_used, converged
                    );
                    // std::cout << "(check) sample[0] trace estimate: " << samples[i][0] << std::endl;
                } // end critical section
            }
        }

        // Elapsed wall time of the algorithm (computation only, not array i/o)
        timer.stop();
        alg_wall_time = timer.elapsed();

        // std::cout << "timer ended: averaging estimates" << std::endl;
        // Remove outliers from trace estimates and average trace estimates
        // std::cout << "samples[0][0]" << samples[0][0] << std::endl;
        // TODO: make this ignore nan's! or figure out a way of skipping if nans detected... or remove nans...
        ConvergenceTools<DataType>::average_estimates(
            confidence_level, outlier_significance_level, num_parameters,
            max_num_samples, num_samples_used, processed_samples_indices,
            samples, num_outliers, trace, error
        );
        // std::cout << "samples[0][0]" << samples[0][0] << std::endl;

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

    template < bool gramian, std::floating_point DataType, LinearOperator Matrix, std::invocable< DataType > Func, ThreadSafeRBG RBG >
    void stochastic_lanczos_quadrature(
        Matrix* A,
        const DataType* parameters,
        const IndexType num_parameters,
        Func&& matrix_function,
        const FlagType orthogonalize,
        const IndexType lanczos_degree,
        const DataType lanczos_tol,
        RBG& rng,
        const IndexType distr,
        DataType* random_vector,
        FlagType* converged,
        DataType* trace_estimate)
    {
        IndexType matrix_size = static_cast< std::pair< size_t, size_t > >(A->shape()).first;

        // Fill random vectors with Rademacher distribution (+1, -1), normalized
        // but not orthogonalized. Settng num_threads to zero indicates to not
        // create any new threads in RandomNumbrGenerator since the current
        // function is inside a parallel thread.
        // std::cout << "Generating random vector" << std::endl;
        IndexType num_threads = 0;
        switch(distr){
            case 0: // rademacher
                generate_array< 0, DataType >(rng, random_vector, matrix_size, num_threads);
                break; 
            default: // normal 
                generate_array< 1, DataType >(rng, random_vector, matrix_size, num_threads);
                break;
        }
        
        // Allocate diagonals (alpha) and supdiagonals (beta) of Lanczos matrix
        DataType* alpha = new DataType[lanczos_degree];
        DataType* beta = new DataType[lanczos_degree];
        for (IndexType cc=0; cc < lanczos_degree; ++cc) {
            alpha[cc] = beta[cc] = 0.0; // this didn't help with nan's...
        }

        // Define 2D arrays needed to decomposition. All these arrays are
        // defined as 1D array with Fortran ordering
        DataType* eigenvectors = NULL;
        DataType* left_singularvectors = NULL;
        DataType* right_singularvectors_transposed = NULL;

        IndexType required_num_inquiries = num_parameters;
        if constexpr(AffineOperator< Matrix >){
            // Special case where, given eig(A), the values eig(A + tB) are known for any t
            required_num_inquiries = Matrix::relation_known ? 1 : num_parameters; 
        }
        // std::cout << "Setting req. # inqueries to " << required_num_inquiries << std::endl;

        // Allocate and initialize theta
        IndexType i, j;
        DataType** theta = new DataType*[num_parameters];
        DataType** tau = new DataType*[num_parameters];
        for (j=0; j < num_parameters; ++j) {
            theta[j] = new DataType[lanczos_degree];
            tau[j] = new DataType[lanczos_degree];
            for (i=0; i < lanczos_degree; ++i) {
                theta[j][i] = 0.0;  
                tau[j][i] = 0.0;   
            }
        }

        // Non-zero size of the tri-diagonal (or bi-diagonal) matrix. Ideally, this matrix
        // is of the size lanczos_degree. But, due to the early termination, it might be smaller.
        IndexType* lanczos_size = new IndexType[num_parameters];

        // MJP: Choosing between gramian / bidiagonal matrix should be zero-cost, so we use constexpr!
        if constexpr(!gramian) {
            // Allocate eigenvectors matrix (1D array with Fortran ordering)
            // MJP: Moved eigenvector to a pre-allocation model of size (lanczos_degree * lanczos_degree) 
            // The lanczos_size[j] should never exceed lanczos_degree, and given to eigh_tridiagonal should be safe
            // eigenvectors = new DataType[lanczos_size[j] * lanczos_size[j]];
            // std::cout << "Proceeeding with sampling" << std::endl;
            eigenvectors = new DataType[lanczos_degree * lanczos_degree];
            for (j=0; j < required_num_inquiries; ++j) {
                
                // std::cout << "Setting affine parameter" << std::endl;
                if constexpr (AffineOperator< Matrix >){
                    A->set_parameter(parameters[j]);
                }

                // Triadiagonalizes A into output arrays 'alpha' (diagonals) and 'beta' (subdiagonals)
                // std::cout << "Tridiagonalizing" << std::endl;
                lanczos_size[j] = lanczos_tridiagonalization< DataType >(
                    A, random_vector, matrix_size, lanczos_degree, lanczos_tol, orthogonalize, 
                    alpha, beta
                );
                // std::cout << "Lanczos size: " << lanczos_size[j] << std::endl;

                // Note: alpha is written in-place with eigenvalues
                // std::cout << "Eigen-decomposing tridiagonal" << std::endl;
                eigh_tridiagonal< DataType >(
                    alpha, beta, eigenvectors, lanczos_size[j]
                );
                // std::cout << "Alpha: " << alpha[0] << ", " << alpha[1] << ", " << alpha[2] << ", ..." << std::endl;
                // std::cout << "Beta: " << beta[0] << ", " << beta[1] << ", " << beta[2] << ", ..." << std::endl;

                // theta and tau from singular values and vectors
                // std::cout << "Computing the quadrature rule" << std::endl;
                for (i=0; i < lanczos_size[j]; ++i) {
                    theta[j][i] = alpha[i];
                    tau[j][i] = eigenvectors[i * lanczos_size[j]];
                }
            }
        } else {
            // Use Golub Kahan Bidiagonalization
            static_assert(AdjointOperator< Matrix >);
            for (j=0; j < required_num_inquiries; ++j) {
                // If trace is already converged, do not compute on the new sample.
                // However, exclude the case where required_num_inquiries is not the
                // same as num_inquiries, since in this case, we compute one inquiry
                // for multiple parameters.
                if ((converged[j] == 1) && (required_num_inquiries == num_parameters)) {
                    continue; // MJP: why isn't this a break?
                }

                // Set parameter of linear operator A
                if constexpr (AffineOperator< Matrix >){
                    // I don't understand why an address was passed; also isn't parameters just a single dimensional array?
                    // Why is j multiplied by num_parameters?
                    // A->set_parameters(&parameters[j*num_parameters]); 
                    A->set_parameter(parameters[j]); 
                }

                // Use Golub-Kahn-Lanczos Bi-diagonalization
                // std::cout << "Bidiagonalizing" << std::endl;
                lanczos_size[j] = golub_kahn_bidiagonalization< DataType >(
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
                // std::cout << "Alpha: " << alpha[0] << ", " << alpha[1] << ", " << alpha[2] << ", ..." << std::endl;
                // std::cout << "Beta: " << beta[0] << ", " << beta[1] << ", " << beta[2] << ", ..." << std::endl;

                // theta and tau from singular values and vectors
                for (i=0; i < lanczos_size[j]; ++i) {
                    theta[j][i] = alpha[i] * alpha[i];
                    tau[j][i] = right_singularvectors_transposed[i];
                }
            }
        }

        // Estimate trace using quadrature method
        DataType quadrature_sum;
        for (j=0; j < num_parameters; ++j) {
            // If the j-th inquiry is already converged, skip.
            if (converged[j] == 1) { continue; }

            // Initialize sum for the integral of quadrature
            quadrature_sum = 0.0;

            // Important: This loop should iterate till lanczos_size[j], not lanczos_degree. 
            // Otherwise, if the input matrix is identity, or rank deficient, the computation is wrong. 
            for (i=0; i < lanczos_size[j]; ++i) {
                quadrature_sum += tau[j][i] * tau[j][i] * matrix_function(theta[j][i]);
            }
            // std::cout << "quad sum2: " << quadrature_sum << ", tau[0][0]: " << tau[0][0] << ", theta: " << theta[0][0] << std::endl; 
            // std::cout << "matrix size: " << matrix_size << ", tr est: " << matrix_size * quadrature_sum << std::endl; 
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