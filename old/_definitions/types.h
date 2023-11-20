/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEFINITIONS_TYPES_H_
#define _DEFINITIONS_TYPES_H_

// ========
// Includes
// ========

#include "./definitions.h"

// =====
// Types
// =====

/// Use \c LongIndexType type for long indices where parallelization could be
/// important. This could include, for instance, indices of long columns of
/// matrices, but not short rows.
///
/// This type is intended to be set as <tt>long</tt>. However, because the
/// indices of \c scipy.sparse matrices are stored as \c int (and not
/// \c long) here, a fused type is used to accommodate both \c int and
/// \c long.
///
/// The type of indices of sparse arrays can be cast, for instance by:
///
///     // In this file:
///     ctypedef long IndexType
///
///     // In linear_operator.pyx:LinearOperator:__cinit__()
///     // Add .astype('uint64') to these variables:
///     self.A_indices = A.indices.astype('uint64')
///     self.B_indices = A.indices.astype('uint64')
///     self.A_index_pointer = A.indptr.astype('uint64')
///     self.B_index_pointer = B.indptr.astype('uint64')
///
/// In the above, \c uint64 is equivalent to <tt>long</tt>. Note, this
/// will *copy* the data, since scipy's sparse indices should be casted from
/// \c uint32.

#if (LONG_INT == 1)
    #if (UNSIGNED_LONG_INT == 1)
        typedef unsigned long int LongIndexType;
    #else
        typedef long int LongIndexType;
    #endif
#else
    #if (UNSIGNED_LONG_INT == 1)
        typedef unsigned int LongIndexType;
    #else
        typedef int LongIndexType;
    #endif
#endif

// Used for indices of short row of matrices
typedef int IndexType;

// Used for both flags and integers, including negative integers
typedef int FlagType;

#endif  // _DEFINITIONS_TYPES_H_
