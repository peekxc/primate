/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>, augmented by Matt Piekenbrock <matt.piekenbrock@gmail.com>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DIAGONALIZATION_H_
#define _DIAGONALIZATION_H_

// =======
// Headers
// =======

#include "_definitions/types.h"      // IndexType
#include <cassert>                      // assert
#include <cstdlib>                      // NULL
#include "_lapack/lapack_api.h"      // lapack_xstev, lapack_xbdsdc
// #include <cblas.h>
// #include <mkl.h>

// MJP Added: just link directly to the externs, let meson handle lapack linking
// https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
// needs: _dstev, _sstev, _dbdsdc, _sbdsdc
// extern "C" {
// 	 void _dstev(char* jobz, int* n, double* d, double* e, double* z, int* ldz, double* work, int* info);
//     void _sstev(char* jobz, int* n, float* d, float* e, float* z, int* ldz, float* work, int* info);
//     void _dbdsdc(char* uplo, char* compq, int* n, double* d,
//                  double *e, double* u, int* ldu, double* vt, int* ldvt,
//                  double* q, int* iq, double* work, int* iwork,
//                  int* info);
//     void _sbdsdc(char* uplo, char* compq, int* n, float* d, float *e,
//                  float* u, int* ldu, float* vt, int* ldvt, float* q,
//                  int* iq, float* work, int* iwork, int* info);
// }



// ================
// eigh tridiagonal
// ================

/// \brief         Computes all eigenvalues and eigenvectors of a real and
///                symmetric tridiagonal matrix.
///
/// \details       The symmetric tridiagonal matrix \f$ \mathbf{A} \f$ is
///                decomposed in to:
///                \f[
///                    \mathbf{A} =
///                    \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^{\intercal}
///                \f]
///
///                where \f$ \mathbf{V} \f$ is unitary and \f$
///                \boldsymbol{\Lambda} \f$ is diagonal.
///
///                #### Algorithm
///
///                This function is equivalent to
///                \c scipy.linalg.eigh_tridigonal which wraps around LAPACK's
///                \c sstev and \c dstev subroutine. Except, this function
///                does not acquire python's GIL, whereas the scipy's function
///                does.
///
///                #### References
///
///                *Note:* Remove blank in the URLs below when opened in the
///                browser.
///
///                * LAPACK's dstev routine user guide
///                  http://www.netlib.org/lapack/explore-html/dc/dd2/group__Da
///                  taType_o_t_h_e_reigen_gaaa6df51cfd92c4ab08d41a54bf05c3ab.h
///                  tml
///                * Routines for BLAS, LAPACK, MAGMA
///                  http://www.icl.utk.edu/~mgates3/docs/lapack.html
///                * Comparison of bdsdc and bdsqr methods
///                   http://www.netlib.org/lapack/lug/node53.html
///                * Scipy.linalg.dstev
///                  https://docs.scipy.org/doc/scipy/reference/generated/scipy
///                  .linalg.lapack.dstev.html
///
/// \param[in,out] diagonals
///                A 1D array of the length \c matrix_size containing the
///                diagonal elements of the matrix. This array will be written
///                in-place with the computed eigenvalues. This array is both
///                input and output of this function.
/// \param[in]     subdiagonals
///                A 1D array of the length \c matrix_size-1 containing the
///                sub-diagonal elements of the matrix. This array will be
///                written in-place with intermediate computation values, but
///                is not an output of this function.
/// \param[out]    eigenvectors
///                1D array of size \c matrix_size*matrix_size and represents a
///                2D array of the shape \c (matrix_size,matrix_size). The
///                second index of the equivalent 2D array iterates over the
///                column vectors. The array has Fortran ordering, meaning that
///                the first index is contiguous. Thus the i-th element of the
///                j-th vector should be accessed by
///                \c eigenvectors[j*matrix_size+i]. This array is written
///                in-place and is the output of this function.
/// \param[in]     matrix_size
///                The size of square matrix.
/// \return        The \c info result of the \c sstev and \c dstev subroutine.
///                Zero indicates a successful computation.

template <typename DataType>
int eigh_tridiagonal(
				DataType* diagonals,
				DataType* subdiagonals,
				DataType* eigenvectors,
				IndexType matrix_size) 
{
		char jobz = 'V';                                  // 'V' computes both eigenvalues and eigenvectors
		DataType* work = new DataType[2*matrix_size - 2]; // Workspace array
		int ldz = matrix_size;                            // Leading dimension of 2D eigenvectors array
		int n = static_cast<int>(matrix_size);            // matrix size
		int info;                                         // Error code output

		// Calling Fortran subroutine
		lapack_xstev(&jobz, &n, diagonals, subdiagonals, eigenvectors, &ldz, work, &info);
		
		// Cleanup 
		delete[] work;
		assert((info == 0, "?stev subroutine returned non-zero status."));
		return info;
}


// ==============
// svd bidiagonal
// ==============

/// \brief         Computes all singular-values and left and right eigenvectors
///                of a real and symmetric upper bi-diagonal matrix.
///
/// \details       The symmetric upper bi-diagonal matrix \f$ \mathbf{A} \f$ is
///                decomposed in to
///                \f[
///                    \mathbf{A} =
///                    \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{\intercal}
///                \f]
///                where \f$ \mathbf{U} \f$ and \f$ \mathbf{V} \f$ are unitary
///                and \f$ \boldsymbol{\Sigma} \f$ is diagonal with positive
///                entries.
///
///                #### Algorithm
///
///                This function uses LAPACK's \c dbdsdc Fortran subroutine.
///
///                #### References
///
///                *Note:* Remove blank in the URLs below when opened in the
///                browser.
///
///                * LAPACK's dbdsdc routine user guide
///                  http://www.netlib.org/lapack/explore-html/d9/d08/dbdsdc_8f
///                  .html
///                * Routines for BLAS, LAPACK, MAGMA
///                  http://www.icl.utk.edu/~mgates3/docs/lapack.html>
///                * Comparison of bdsdc and bdsqr methods
///                  http://www.netlib.org/lapack/lug/node53.html
///
/// \param[in, out] diagonals
///                 A 1D array of the length \c matrix_size containing the
///                 diagonal elements of the matrix. This array will be
///                 written in-place with the computed eigenvalues. This array
///                 is both input and output of this function.
/// \param[in]      supdiagonals
///                 A 1D array of the length \c matrix_size-1 containing the
///                 sub-diagonal elements of the matrix. This array will be
///                 written in-place with intermediate computation values, but
///                 is not an output of this function.
/// \param[out]     U
///                 Right eigenvectors represented by 1D array of the length
///                 \c matrix_size*matrix_size which denotes a 2D array of the
///                 shape \c (matrix_size, matrix_size). The second index of
///                 the matrix iterates over the column vectors. The array has
///                 Fortran ordering, meaning that the first index is
///                 contiguous. Thus, the i-th element of the j-th vector
///                 should be accessed by \c U[j*matrix_size+i]. This array is
///                 written in place and is the output of this function.
/// \param[out]     Vt
///                 Transpose of left eigenvectors represented by 1D array of
///                 the length \c matrix_size*matrix_size which denotes a 2D
///                 array of the shape \c (matrix_size, matrix_size). The
///                 second index of the matrix iterates over the column
///                 vectors. The array has Fortran ordering, meaning that the
///                 first index is contiguous. Thus, the i-th element of the
///                 j-th vector should be accessed by \c Vt[j*matrix_size+i].
///                 This array is written in place and is the output of this
///                 function.
/// \param[in]      matrix_size
///                 The size of square matrix.
/// \return         Integer \c info indicates the status of the computation.
///                 Zero indicates a successful computation.

template <typename DataType>
int svd_bidiagonal(
				DataType* diagonals,
				DataType* supdiagonals,
				DataType* U,
				DataType* Vt,
				IndexType matrix_size)
{
		// Code 'U' indicates the matrix is upper bi-diagonal
		char UPLO = 'U';

		// Code 'I' computes both singular values and singular vectors
		char COMPQ = 'I';

		// matrix size
		int n = static_cast<int>(matrix_size);

		// Leading dimensions of arrays U and Vt
		int LDU = matrix_size;
		int LDVT = matrix_size;

		// There arrays are not referenced when COMPQ is set to 'I'
		DataType* Q = NULL;
		int* IQ = NULL;

		// Work arrays
		DataType* work = new DataType[3*matrix_size*matrix_size + 4*matrix_size];
		int* iwork = new int[8 * matrix_size];

		// Error code output
		int info;

		// Calling Fortran subroutine
		lapack_xbdsdc(&UPLO, &COMPQ, &n, diagonals, supdiagonals, U, &LDU, Vt,
						&LDVT, Q, IQ, work, iwork, &info);

		delete[] work;
		delete[] iwork;

		assert((info == 0, "?stev subroutine returned non-zero status."));

		return info;
}

#endif  // _DIAGONALIZATION_H_

// ===============================
// Explicit template instantiation
// ===============================

// template eigh_tridiagonal<float>;
// template svd_bidiagonal<float>;
