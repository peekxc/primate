/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_TRACE_ESTIMATOR_CONVERGENCE_TOOLS_H_
#define _C_TRACE_ESTIMATOR_CONVERGENCE_TOOLS_H_

// ======
// Headers
// ======

#include "../_definitions/types.h"  // IndexType, FlagType


// =================
// c Trace Estimator
// =================

/// \class ConvergenceTools
///
/// \brief A static class to compute the trace of implicit matrix functions
///        using stochastic Lanczos quadrature method. This class acts as a
///        templated namespace, where the member methods is *public* and
///        *static*. The internal private member functions are also static.
///
/// \sa    Diagonalization

template <typename DataType>
class ConvergenceTools
{
    public:

        // _check convergence
        static FlagType check_convergence(
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
                FlagType* converged);

        // _average estimates
        static void average_estimates(
                const DataType confidence_level,
                const DataType outlier_significance_level,
                const IndexType num_inquiries,
                const IndexType max_num_samples,
                const IndexType* num_samples_used,
                const IndexType* processed_samples_indices,
                DataType** samples,
                IndexType* num_outliers,
                DataType* trace,
                DataType* error);
};

#endif  // _C_TRACE_ESTIMATOR_CONVERGENCE_TOOLS_H_
