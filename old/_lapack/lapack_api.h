/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _LAPACK_API_H_
#define _LAPACK_API_H_

extern "C" {
    void dstev(const char* jobz, const int* n, double* d, double* e,
            double* z, const int* ldz, double* work, int* info) noexcept;
    void sstev(const char* jobz, const int* n, float* d, float* e, float* z,
            const int* ldz, float* work, int* info) noexcept;
    void dbdsdc(const char* uplo, const char* compq, const int* n, double* d,
                 double *e, double* u, const int* ldu, double* vt, const int* ldvt,
                 double* q, int* iq, double* work, int* iwork,
                 int* info) noexcept;
    void sbdsdc(const char* uplo, const char* compq, const int* n, float* d,
             float* e, float* u, const int* ldu, float* vt,
             const int* ldvt, float* q, int* iq, float* work,
             int* iwork, int* info) noexcept;
}

template <typename DataType>
void lapack_xstev(char* jobz, int* n, DataType* d, DataType* e, DataType* z,
        int* ldz, DataType* work, int* info);

template <typename DataType>
void lapack_xbdsdc(char* uplo, char* compq, int* n, DataType* d, DataType *e,
        DataType* u, int* ldu, DataType* vt, int* ldvt, DataType* q,
        int* iq, DataType* work, int* iwork, int* info);


#include "lapack_api.hpp"

#endif  // _LAPACK_API_H_
