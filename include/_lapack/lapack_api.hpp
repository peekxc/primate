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

#include <cstddef>  // NULL


// ============
// lapack xstev (float specialization)
// ============

/// \brief Overlodng wrapper for both \c lapack_sstev (a float function) and
///        \c lapack_dstev (a double function). \c xstev overloads both
///        \c sstev and \c dstev with the same function signature.

template<>
void lapack_xstev<float>(char* jobz, int* n, float* d, float* e, float* z,
                         int* ldz, float* work, int* info)
{
    // Calling float method
    sstev(jobz, n, d, e, z, ldz, work, info);
}


// ============
// lapack xstev (double specialization)
// ============

/// \brief Overlodng wrapper for both \c lapack_sstev (a float function) and
///        \c lapack_dstev (a double function). \c xstev overloads both
///        \c sstev and \c dstev with the same function signature.

template<>
void lapack_xstev<double>(char* jobz, int* n, double* d, double* e, double* z,
                          int* ldz, double* work, int* info)
{
    // Calling double method
    dstev(jobz, n, d, e, z, ldz, work, info);
}

// ============
// lapack xstev (long double specialization)
// ============

/// \brief   Overlodng wrapper for both \c lapack_sstev (a float function) and
///          \c lapack_dstev (a double function). \c xstev overloads both
///          \c sstev and \c dstev with the same function signature. This
///          function casts `long double` type to \c double and uses \c dstev
///          subroutine.
///
/// \details The variables with leading undescore are \c double counterparts
///          of the `long double` variables.

template<>
void lapack_xstev<long double>(char* jobz, int* n, long double* d,
                               long double* e, long double* z, int* ldz,
                               long double* work, int* info)
{
    // Mark unused variables to avoid compiler warnings (-Wno-unused-parameter)
    (void) work;

    // Deep copy long double diagonal array to double
    double *d_ = new double[(*n)];
    for (int i=0; i < (*n); ++i)
    {
        d_[i] = static_cast<double>(d[i]);
    }

    // Deep copy long double supdiagonal array to double
    double *e_ = new double[(*n)-1];
    for (int i=0; i < (*n)-1; ++i)
    {
        e_[i] = static_cast<double>(e[i]);
    }

    // Declare eigenvectors and work arrays as double
    double *z_ = new double[(*ldz)*(*n)];
    double *work_ = new double[2*(*n)-2];

    // Calling double method
    dstev(jobz, n, d_, e_, z_, ldz, work_, info);

    // Copy eigenvalues from double to long double
    for (int i=0; i < (*n); ++i)
    {
        d[i] = static_cast<long double>(d_[i]);
    }

    // Copy eigenvectors from double to long double
    for (int i=0; i < (*ldz)*(*n); ++i)
    {
        z[i] = static_cast<long double>(z_[i]);
    }

    // Deallocate memory
    delete[] d_;
    delete[] e_;
    delete[] z_;
    delete[] work_;
}


// =============
// lapack xbdsdc (float specialization)
// =============

/// \brief Overlodng wrapper for both \c lapack_sbdsdc (a float function) and
///        \c lapack_dbdsdc (a double function). \c xbdsdc overloads both
///        \c sbdsdc and \c dbdsdc with the same function signature.

template<>
void lapack_xbdsdc<float>(char* uplo, char* compq, int* n, float* d, float *e,
                          float* u, int* ldu, float* vt, int* ldvt, float* q,
                          int* iq, float* work, int* iwork, int* info)
{
    sbdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork, info);
}


// =============
// lapack xbdsdc (double specialization)
// =============

/// \brief Overlodng wrapper for both \c lapack_sbdsdc (a double function) and
///        \c lapack_dbdsdc (a double function). \c xbdsdc overloads both
///        \c sbdsdc and \c dbdsdc with the same function signature.

template<>
void lapack_xbdsdc<double>(char* uplo, char* compq, int* n, double* d,
                           double *e, double* u, int* ldu, double* vt,
                           int* ldvt, double* q, int* iq, double* work,
                           int* iwork, int* info)
{
    dbdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork, info);
}


// =============
// lapack xbdsdc (long double specialization)
// =============

/// \brief   Overlodng wrapper for both \c lapack_sbdsdc (a double function)
///          and \c lapack_dbdsdc (a double function). \c xbdsdc overloads both
///          \c sstev and \c dstev with the same function signature. This
///          function casts `long double` type to \c double and uses \c dstev
///          subroutine.
///
/// \details The variables with leading undescore are \c double counterparts
///          of the `long double` variables.

template<>
void lapack_xbdsdc<long double>(char* uplo, char* compq, int* n,
                                long double* d, long double *e, long double* u,
                                int* ldu, long double* vt, int* ldvt,
                                long double* q, int* iq, long double* work,
                                int* iwork, int* info)
{
    // Mark unused variables to avoid compiler warnings (-Wno-unused-parameter)
    (void) q;
    (void) work;

    // Deep copy long double diagonal array to double
    double *d_ = new double[(*n)];
    for (int i=0; i < (*n); ++i)
    {
        d_[i] = static_cast<double>(d[i]);
    }

    // Deep copy long double supdiagonal array to double
    double *e_ = new double[(*n)-1];
    for (int i=0; i < (*n)-1; ++i)
    {
        e_[i] = static_cast<double>(e[i]);
    }

    // Declare left and right eigenvectors arrays
    double *u_ = new double[(*ldu)*(*n)];
    double *vt_ = new double[(*ldvt)*(*n)];

    // Declare work variables
    double* q_ = NULL;
    double *work_ = new double[3*(*n)*(*n) + 4*(*n)];

    // Call lapack
    dbdsdc(uplo, compq, n, d_, e_, u_, ldu, vt_, ldvt, q_, iq, work_, iwork, info);

    // Copy back eigenvectors from double to long double
    for (int i=0; i < (*n); ++i)
    {
        d[i] = static_cast<long double>(d_[i]);
    }

    // Copy left and right eigenvectors fom double to long double
    for (int i=0; i < (*ldu)*(*n); ++i)
    {
        u[i] = static_cast<long double>(u_[i]);
    }

    for (int i=0; i < (*ldvt)*(*n); ++i)
    {
        vt[i] = static_cast<long double>(vt_[i]);
    }

    // Deallocate memory
    delete[] d_;
    delete[] e_;
    delete[] u_;
    delete[] vt_;
    delete[] work_;
}
