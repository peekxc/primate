/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./convergence_tools.h"
#include <cmath>  // sqrt, std::abs, INFINITY, NAN, isnan
#include <algorithm>  // std::max
#include "./special_functions.h"  // erf_inv


// =================
// check convergence
// =================

/// \brief      Checks if the standard deviation of the set of the cumulative
///             averages of trace estimators converged below the given
///             tolerance.
///
/// \details    The convergence criterion for each trace inquiry is if:
///
///                 standard_deviation < max(rtol * average[i], atol)
///
///             where \c rtol and \c atol are relative and absolute tolerances,
///             respectively. If this criterion is satisfied for *all* trace
///             inquiries, this function returns \c 1, otherwise \c 0.
///
/// \param[in]  samples
///             A 2D array of the estimated trace. This array has the shape
///             \c (max_num_samples,num_inquiries). Each column of this array
///             is the estimated trace values for a different trace inquiry.
///             The rows are Monte-Carlo samples of the estimation of trace.
///             The cumulative average of the rows are expected to converge.
///             Note that because of parallel processing, the rows of this
///             array are not filled sequentially. The list of filled rows are
///             stored in \c processed_samples_indices.
/// \param[in]  min_num_samples
///             Minimum number of sample iterations.
/// \param[in]  num_inquiries
///             Number of columns of \c samples array.
/// \param[in]  processed_samples_indices
///             A 1D array indicating the processing order of rows of the
///             \c samples. In parallel processing, this order of processing
///             the rows of \c samples is not necessarly sequential.
/// \param[in]  num_processed_samples
///             A counter that keeps track of how many samples were processed
///             so far in the iterations.
/// \param[in]  confidence_level
///             The confidence level of the error, which is a number between
///             \c 0 and \c 1. This affects the scale of \c error.
/// \param[in]  error_atol
///             Absolute tolerance criterion to be compared with the standard
///             deviation of the cumulative averages. If \c error_atol is zero,
///             it is ignored, and only \c rtol is used.
/// \param[in]  error_rtol
///             Relative tolerance criterion to be compared with the standard
///             deviation of the cumulative averages. If \c error_rtol is zero,
///             it is ignored, and only \c error_atol is used.
/// \param[out] error
///             The error of estimation of trace, which is the standard
///             deviation of the rows of \c samples array. The size of this
///             array is \c num_inquiries.
/// \param[out] num_samples_used
///             1D array of the size of the number of columns of \c samples.
///             Each element indicates how many iterations were used till
///             convergence is reached for each column of the \c samples. The
///             number of iterations should be a number between
///             \c min_num_samples and \c max_num_samples.
/// \param[out] converged
///             1D array of the size of the number of columns of \c samples.
///             Each element indicates which column of \c samples has converged
///             to the tolerance criteria. Normally, if the \c num_samples_used
///             is less than \c max_num_samples, it indicates that the
///             convergence has reached. the rows of \c samples array. The size
///             of this array is \c num_inquiries.
/// \return     A signal to indicate the status of computation:
///             * \c 1 indicates successful convergence within the given
///               tolerances was met. Convergence is achieved when all elements
///               of \c convergence array are below \c convergence_atol or
///               \c convergence_rtol times \c trace.
///             * \c 0 indicates the convergence criterion was not met for at
///               least one of the trace inquiries.

template <typename DataType>
FlagType ConvergenceTools<DataType>::check_convergence(
        DataType** samples,
        const IndexType min_num_samples,
        const IndexType num_inquiries,
        const IndexType* processed_samples_indices,
        const IndexType num_processed_samples,
        const DataType confidence_level,
        const DataType error_atol,
        const DataType error_rtol,
        DataType* error,
        IndexType* num_samples_used,
        FlagType* converged)
{
    FlagType all_converged;
    IndexType j;

    // If number of processed samples are not enough, set to not converged yet.
    // This is essential since in the first few iterations, the standard
    // deviation of the cumulative averages are still too small.
    if (num_processed_samples < min_num_samples)
    {
        // Skip computing error. Fill outputs with trivial initial values
        for (j=0; j < num_inquiries; j++)
        {
            error[j] = INFINITY;
            converged[j] = 0;
            num_samples_used[j] = num_processed_samples;
        }
        all_converged = 0;
        return all_converged;
    }

    IndexType i;
    DataType summand;
    DataType mean;
    DataType std;
    DataType data;

    // Quantile of normal distribution (usually known as the "z" coefficient)
    DataType standard_z_score = sqrt(2) * \
        static_cast<DataType>(erf_inv(static_cast<double>(confidence_level)));

    // For each column of samples, compute error of all processed rows
    for (j=0; j < num_inquiries; ++j)
    {
        // Do not check convergence if j-th column already converged
        if (converged[j] == 0)
        {
            // mean of j-th column using all processed rows of j-th column
            summand = 0.0;
            for (i=0; i < num_processed_samples; ++i)
            {
                summand += samples[processed_samples_indices[i]][j];
            }
            mean = summand / num_processed_samples;

            // std of j-th column using all processed rows of j-th column
            if (num_processed_samples > 1)
            {
                summand = 0.0;
                for (i=0; i < num_processed_samples; ++i)
                {
                    data = samples[processed_samples_indices[i]][j];
                    summand += (data - mean) * (data - mean);
                }
                std = sqrt(summand / (num_processed_samples - 1.0));
            }
            else
            {
                std = INFINITY;
            }

            // Compute error based of std and confidence level
            error[j] = standard_z_score * std / sqrt(num_processed_samples);

            // Check error with atol and rtol to find if j-th column converged
            if (error[j] < std::max(error_atol, error_rtol*mean))
            {
                converged[j] = 1;
            }

            // Update how many samples used so far to average j-th column
            num_samples_used[j] = num_processed_samples;
        }
    }

    // Check convergence is reached for all columns (all inquiries)
    all_converged = 1;
    for (j=0; j < num_inquiries; ++j)
    {
        if (converged[j] == 0)
        {
            // The j-th column not converged.
            all_converged = 0;
            break;
        }
    }

    return all_converged;
}


// =================
// average estimates
// =================

/// \brief      Averages the estimates of trace. Removes outliers and
///             reevaluates the error to take into account for the removal of
///             the outliers.
///
/// \note       The elimination of outliers does not affect the elements of
///             samples array, rather it only affects the reevaluation of trac
///             and error arrays.
///
/// \param[in]  confidence_level
///             The confidence level of the error, which is a number between
///             \c 0 and \c 1. This affects the scale of \c error.
/// \param[in]  outlier_significance_level
///             One minus the confidence level of the uncertainty band of the
///             outlier. This is a number between \c 0 and \c 1. Confidence
///             level of outleir and significance level of outlier are
///             commlement of each other.
/// \param[in]  num_inquiries
///             The number of batches of parameters. This function computes
///             \c num_inquiries values of trace corresponding to different
///             batch of parameters of the linear operator \c A. Hence, the
///             number of output trace is \c num_inquiries. Hence, it is the
///             number of columns of the output array \c samples.
/// \param[in]  max_num_samples
///             The number of times that the trace estimation is repeated. The
///             output trace value is the average of the samples. Hence, this
///             is the number of rows of the output array \c samples. Larger
///             number of samples leads to a better trace estimation. The
///             computational const linearly increases with number of samples.
/// \param[in]  num_samples_used
///             1D array of the size of the number of columns of \c samples.
///             Each element indicates how many iterations were used till
///             convergence is reached for each column of the \c samples. The
///             number of iterations should be a number between
///             \c min_num_samples and \c max_num_samples.
/// \param[in]  processed_samples_indices
///             A 1D array indicating the processing order of rows of the
///             \c samples. In paralleli processing, this order of processing
///             the rows of \c samples is not necessarly sequential.
/// \param[out] samples
///             2D array of all estimated trace samples. The shape of this
///             array is \c (max_num_samples*num_inquiries). The average of the
///             rows is also given in \c trace array.
/// \param[out] num_outliers
///             1D array with the size of number of columns of \c samples. Each
///             element indicates how many rows of the \c samples array were
///             outliers and were removed during averaging rows of \c samples.
/// \param[out] trace
///             The output trace of size \c num_inquiries. These values are the
///             average of the rows of \c samples array.
/// \param[out] error
///             The error of estimation of trace, which is the standard
///             deviation of the rows of \c samples array. The size of this
///             array is \c num_inquiries.

template <typename DataType>
void ConvergenceTools<DataType>::average_estimates(
        const DataType confidence_level,
        const DataType outlier_significance_level,
        const IndexType num_inquiries,
        const IndexType max_num_samples,
        const IndexType* num_samples_used,
        const IndexType* processed_samples_indices,
        DataType** samples,
        IndexType* num_outliers,
        DataType* trace,
        DataType* error)
{
    IndexType i;
    IndexType j;
    DataType summand;
    DataType mean;
    DataType std;
    DataType mean_discrepancy;
    DataType outlier_half_interval;

    // Flag which samples are outliers
    FlagType* outlier_indices = new FlagType[max_num_samples];

    // Quantile of normal distribution (usually known as the "z" coefficient)
    DataType error_z_score = sqrt(2) * erf_inv(confidence_level);

    // Confidence level of outlier is the complement of significance level
    DataType outlier_confidence_level = 1.0 - outlier_significance_level;

    // Quantile of normal distribution area where is not considered as outlier
    DataType outlier_z_score = sqrt(2.0) * erf_inv(outlier_confidence_level);

    for (j=0; j < num_inquiries; ++j)
    {
        // Initialize outlier indices for each column of samples
        for (i=0; i < max_num_samples; ++i)
        {
            outlier_indices[i] = 0;
        }
        num_outliers[j] = 0;

        // Compute mean of the j-th column
        summand = 0.0;
        for (i=0; i < num_samples_used[j]; ++i)
        {
            summand += samples[processed_samples_indices[i]][j];
        }
        mean = summand / num_samples_used[j];

        // Compute std of the j-th column

        if (num_samples_used[j] > 1)
        {
            summand = 0.0;
            for (i=0; i < num_samples_used[j]; ++i)
            {
                mean_discrepancy = \
                    samples[processed_samples_indices[i]][j] - mean;
                summand += mean_discrepancy * mean_discrepancy;
            }
            std = sqrt(summand / (num_samples_used[j] - 1.0));
        }
        else
        {
            std = INFINITY;
        }

        // Outlier half interval
        outlier_half_interval = outlier_z_score * std;

        // Difference of each element from
        for (i=0; i < num_samples_used[j]; ++i)
        {
            mean_discrepancy = samples[processed_samples_indices[i]][j] - mean;
            if (std::abs(mean_discrepancy) > outlier_half_interval)
            {
                // Outlier detected
                outlier_indices[i] = 1;
                num_outliers[j] += 1;
            }
        }

        // Reevaluate mean but leave out outliers
        summand = 0.0;
        for (i=0; i < num_samples_used[j]; ++i)
        {
            if (outlier_indices[i] == 0)
            {
                summand += samples[processed_samples_indices[i]][j];
            }
        }
        mean = summand / (num_samples_used[j] - num_outliers[j]);

        // Reevaluate std but leave out outliers
        if (num_samples_used[j] > 1 + num_outliers[j])
        {
            summand = 0.0;
            for (i=0; i < num_samples_used[j]; ++i)
            {
                if (outlier_indices[i] == 0)
                {
                    mean_discrepancy = \
                        samples[processed_samples_indices[i]][j] - mean;
                    summand += mean_discrepancy * mean_discrepancy;
                }
            }
            std = sqrt(summand/(num_samples_used[j] - num_outliers[j] - 1.0));
        }
        else
        {
            std = INFINITY;
        }

        // trace and its error
        trace[j] = mean;
        error[j] = error_z_score * std / \
            sqrt(num_samples_used[j] - num_outliers[j]);
    }

    delete[] outlier_indices;
}


// ===============================
// Explicit template instantiation
// ===============================

template class ConvergenceTools<float>;
template class ConvergenceTools<double>;
template class ConvergenceTools<long double>;
