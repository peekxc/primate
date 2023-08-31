/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEFINITIONS_DEFINITIONS_H_
#define _DEFINITIONS_DEFINITIONS_H_


// ===========
// Definitions
// ===========

// If set to 0, the LongIndexType is declared as 32-bit integer. Whereas if set
// to 1, the LongIndexType is declared as 64-bit integer. The long integer will
// slow down the performance on reading array if integers. Note that in C++,
// there is no difference between "int" and "long int". That is, both are 32
// bit. To see the real effect of long type, define the integer by "long long"
// rather than "long int". The "long long" is indeed 64-bit. Currently, the
// long type in "./types.h" is defined as "long int". Hence, setting LONG_INT
// to 1 will not make any difference unless "long long" is used.
//
// Note: The malloc and cudaMalloc can only allocate at maximum, an array of
// the limit size of "size_t" (unsigned int). So, using "long long int" is
// not indeed practical for malloc. Thus, it is better to set the type of array
// indices as just "signed int".
#ifndef LONG_INT
    #define LONG_INT 0
#endif

// If set to 0, the LongIndexType is declared as signed integer, whereas if set
// to 1, the LongIndexType is declared as unsigned integer. The unsigned type
// will double the limit of the largest integer index, while keeps the same
// speed for index operations. Note that the indices and index pointers of
// scipy sparse arrays are defined by "signed int". Hence, by setting
// UNSIGNED_LONG_INT to 1, there is a one-time overhead of convening the numpy
// int arrays (two matrices of scipy.sparse.csr_matrix.indices and
// scipy.sparse.csr_matrix.indptr) from "int" to "unsigned int". This overhead
// is only one-time and should be around half a second for moderate to large
// arrays. But, on the positive side, the unsigned int can handle arrays of
// up to twice the index size.
//
// Note: The malloc and cudaMalloc can only allocate at maximum, an array of
// the limit size of "size_t" (unsigned int). So, using "unsigned int" for
// index is not indeed practical since the array size in bytes is the size of
// array times sizeof(DataType). That is, if DataType is double for instance,
// the maximum array size could potentially be 8 times the size of maximum
// of "size_t" (unsigned int) which is not possible for malloc. Thus, it is
// better to set the type of array indices as just "signed int".
#ifndef UNSIGNED_LONG_INT
    #define UNSIGNED_LONG_INT 0
#endif

// If USE_CBLAS is set to 1, the OpenBlas library is used for dense vector and
// matrix operations. Note that Openblas does not declare operations on "long
// double" type, rather, only "float" and "double" types are supported. To use
// "long double" type, set USE_CBLAS to 0. Openblas is nearly twice faster, but
// it looses accuracy on large arrays of float type. This inaccuracy could
// matter a lot when computing dot product and norm of very large vectors.
#ifndef USE_CBLAS
    #define USE_CBLAS 0
#endif


#endif  // _DEFINITIONS_DEFINITIONS_H_
