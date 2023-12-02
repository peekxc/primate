/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _C_BASIC_ALGEBRA_C_MATRIX_OPERATIONS_H_
#define _C_BASIC_ALGEBRA_C_MATRIX_OPERATIONS_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // IndexType, LongIndexType, FlagType
#include "../_definitions/definitions.h"  // USE_CBLAS

#if (USE_CBLAS == 1)
	#include "./cblas_interface.h"
#endif


// ============
// dense matvec
// ============

/// \brief      Computes the matrix vector multiplication \f$ \boldsymbol{c} =
///             \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is a
///             dense matrix.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c
///             float, the summation loses the precision, especially when the
///             vector size is large. It seems that using <tt>long double</tt>
///             is slightly faster than using \c double. The advantage of using
///             a type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
///
/// \param[in]  A
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  b
///             Column vector
/// \param[in]  num_rows
///             Number of rows of \c A
/// \param[in]  num_columns
///             Number of columns of \c A
/// \param[in]  A_is_row_major
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * If \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[out] c
///             The output column vector (written in-place).

template < typename DataType >
struct cMatrixOperations {

	static void dense_matvec(
			const DataType* A,
			const DataType* b,
			const LongIndexType num_rows,
			const LongIndexType num_columns,
			const FlagType A_is_row_major,
			DataType* c)
	{
		#if (USE_CBLAS == 1)

		// Using OpenBlas
		CBLAS_LAYOUT layout;
		if (A_is_row_major)
		{
			layout = CblasRowMajor;
		}
		else
		{
			layout = CblasColMajor;
		}

		CBLAS_TRANSPOSE transpose = CblasNoTrans;
		int lda = num_rows;
		int incb = 1;
		int incc = 1;
		DataType alpha = 1.0;
		DataType beta = 0.0;

		cblas_interface::xgemv(layout, transpose, num_rows, num_columns, alpha, A,
							lda, b, incb, beta, c, incc);

		#else

		// Not using OpenBlas
		LongIndexType j;
		long double sum;
		LongIndexType chunk = 5;
		LongIndexType num_columns_chunked = num_columns - (num_columns % chunk);

		// Determine major order of A
		if (A_is_row_major) {
			// For row major (C ordering) matrix A
			for (LongIndexType i=0; i < num_rows; ++i)
			{
				sum = 0.0;
				for (j=0; j < num_columns_chunked; j+= chunk) {
					sum += A[i*num_columns + j] * b[j] +
						A[i*num_columns + j+1] * b[j+1] +
						A[i*num_columns + j+2] * b[j+2] +
						A[i*num_columns + j+3] * b[j+3] +
						A[i*num_columns + j+4] * b[j+4];
				}

				for (j= num_columns_chunked; j < num_columns; ++j) {
					sum += A[i*num_columns + j] * b[j];
				}

				c[i] = static_cast<DataType>(sum);
			}
		}
		else {
			// For column major (Fortran ordering) matrix A
			for (LongIndexType i=0; i < num_rows; ++i)
			{
				sum = 0.0;
				for (j=0; j < num_columns; ++j){
					sum += A[i + num_rows*j] * b[j];
				}
				c[i] = static_cast<DataType>(sum);
			}
		}

		#endif
	}


// =================
// dense matvec plus
// =================

/// \brief         Computes the operation \f$ \boldsymbol{c} = \boldsymbol{c} +
///                \alpha \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A}
///                \f$ is a dense matrix.
///
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
///
/// \param[in]     A
///                1D array that represents a 2D dense array with either C
///                (row) major ordering or Fortran (column) major ordering. The
///                major ordering should de defined by \c A_is_row_major flag.
/// \param[in]     b
///                Column vector
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of \c A
/// \param[in]     num_columns
///                Number of columns of \c A
/// \param[in]     A_is_row_major
///                Boolean, can be \c 0 or \c 1 as follows:
///                * If \c A is row major (C ordering where the last index is
///                  contiguous) this value should be \c 1.
///                * If \c A is column major (Fortran ordering where the first
///                  index is contiguous), this value should be set to \c 0.
/// \param[in,out] c
///                The output column vector (written in-place).

static void dense_matvec_plus(
		const DataType* A,
		const DataType* b,
		const DataType alpha,
		const LongIndexType num_rows,
		const LongIndexType num_columns,
		const FlagType A_is_row_major,
		DataType* c)
{
	if (alpha == 0.0){
		return;
	}

	LongIndexType j;
	long double sum;
	LongIndexType chunk = 5;
	LongIndexType num_columns_chunked = num_columns - (num_columns % chunk);

	// Determine major order of A
	if (A_is_row_major)
	{
		// For row major (C ordering) matrix A
		for (LongIndexType i=0; i < num_rows; ++i)
		{
			sum = 0.0;
			for (j=0; j < num_columns_chunked; j+= chunk)
			{
				sum += A[i*num_columns + j] * b[j] +
					   A[i*num_columns + j+1] * b[j+1] +
					   A[i*num_columns + j+2] * b[j+2] +
					   A[i*num_columns + j+3] * b[j+3] +
					   A[i*num_columns + j+4] * b[j+4];
			}

			for (j= num_columns_chunked; j < num_columns; ++j)
			{
				sum += A[i*num_columns + j] * b[j];
			}

			c[i] += alpha * static_cast<DataType>(sum);
		}
	}
	else
	{
		// For column major (Fortran ordering) matrix A
		for (LongIndexType i=0; i < num_rows; ++i)
		{
			sum = 0.0;
			for (j=0; j < num_columns; ++j)
			{
				sum += A[i + num_rows*j] * b[j];
			}
			c[i] += alpha* static_cast<DataType>(sum);
		}
	}
}


// =======================
// dense transposed matvec
// =======================

/// \brief      Computes matrix vector multiplication \f$\boldsymbol{c} =
///             \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$ \mathbf{A}
///             \f$ is dense, and \f$ \mathbf{A}^{\intercal} \f$ is the
///             transpose of the matrix \f$ \mathbf{A} \f$.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c float,
///             the summation loses the precision, especially when the vector
///             size is large. It seems that using <tt>long double</tt> is
///             slightly faster than using \c double. The advantage of using a
///             type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
///
/// \param[in]  A
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  b
///             Column vector
/// \param[in]  num_rows
///             Number of rows of \c A
/// \param[in]  num_columns
///             Number of columns of \c A
/// \param[in]  A_is_row_major
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * f \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[out] c
///             The output column vector (written in-place).

static void dense_transposed_matvec(
		const DataType* A,
		const DataType* b,
		const LongIndexType num_rows,
		const LongIndexType num_columns,
		const FlagType A_is_row_major,
		DataType* c)
{
	LongIndexType i;
	long double sum;
	LongIndexType chunk = 5;
	LongIndexType num_rows_chunked = num_rows - (num_rows % chunk);

	// Determine major order of A
	if (A_is_row_major)
	{
		// For row major (C ordering) matrix A
		for (LongIndexType j=0; j < num_columns; ++j)
		{
			sum = 0.0;
			for (i=0; i < num_rows; ++i)
			{
				sum += A[i*num_columns + j] * b[i];
			}
			c[j] = static_cast<DataType>(sum);
		}
	}
	else
	{
		// For column major (Fortran ordering) matrix A
		for (LongIndexType j=0; j < num_columns; ++j)
		{
			sum = 0.0;
			for (i=0; i < num_rows_chunked; i += chunk)
			{
				sum += A[i + num_rows*j] * b[i] +
					   A[i+1 + num_rows*j] * b[i+1] +
					   A[i+2 + num_rows*j] * b[i+2] +
					   A[i+3 + num_rows*j] * b[i+3] +
					   A[i+4 + num_rows*j] * b[i+4];
			}

			for (i=num_rows_chunked; i < num_rows; ++i)
			{
				sum += A[i + num_rows*j] * b[i];
			}

			c[j] = static_cast<DataType>(sum);
		}
	}
}


// ============================
// dense transposed matvec plus
// ============================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$
///                \mathbf{A} \f$ is dense, and \f$ \mathbf{A}^{\intercal} \f$
///                is the transpose of the matrix \f$ \mathbf{A} \f$.
///
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
///
/// \param[in]     A
///                1D array that represents a 2D dense array with either C
///                (row) major ordering or Fortran (column) major ordering. The
///                major ordering should de defined by \c A_is_row_major flag.
/// \param[in]     b
///                Column vector
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of \c A
/// \param[in]     num_columns
///                Number of columns of \c A
/// \param[in]     A_is_row_major
///                Boolean, can be \c 0 or \c 1 as follows:
///                * If \c A is row major (C ordering where the last index is
///                  contiguous) this value should be \c 1.
///                * f \c A is column major (Fortran ordering where the first
///                  index is contiguous), this value should be set to \c 0.
/// \param[in,out] c
///                The output column vector (written in-place).


static void dense_transposed_matvec_plus(
		const DataType* A,
		const DataType* b,
		const DataType alpha,
		const LongIndexType num_rows,
		const LongIndexType num_columns,
		const FlagType A_is_row_major,
		DataType* c)
{
	if (alpha == 0.0)
	{
		return;
	}

	LongIndexType i;
	long double sum;
	LongIndexType chunk = 5;
	LongIndexType num_rows_chunked = num_rows - (num_rows % chunk);

	// Determine major order of A
	if (A_is_row_major)
	{
		// For row major (C ordering) matrix A
		for (LongIndexType j=0; j < num_columns; ++j)
		{
			sum = 0.0;
			for (i=0; i < num_rows; ++i)
			{
				sum += A[i*num_columns + j] * b[i];
			}
			c[j] += alpha * static_cast<DataType>(sum);
		}
	}
	else
	{
		// For column major (Fortran ordering) matrix A
		for (LongIndexType j=0; j < num_columns; ++j)
		{
			sum = 0.0;
			for (i=0; i < num_rows_chunked; i += chunk)
			{
				sum += A[i + num_rows*j] * b[i] +
					   A[i+1 + num_rows*j] * b[i+1] +
					   A[i+2 + num_rows*j] * b[i+2] +
					   A[i+3 + num_rows*j] * b[i+3] +
					   A[i+4 + num_rows*j] * b[i+4];
			}

			for (i=num_rows_chunked; i < num_rows; ++i)
			{
				sum += A[i + num_rows*j] * b[i];
			}

			c[j] += alpha * static_cast<DataType>(sum);
		}
	}
}


// ==========
// csr matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A} \f$ is compressed sparse row (CSR) matrix
///             and \f$ \boldsymbol{b} \f$ is a dense vector. The output \f$
///             \boldsymbol{c} \f$ is a dense vector.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c float,
///             the summation loses the precision, especially when the vector
///             size is large. It seems that using <tt>long double</tt> is
///             slightly faster than using \c double. The advantage of using a
///             type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
///
/// \param[in]  A_data
///             CSR format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_column_indices
///             CSR format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSR format index pointer. The length of this array is one plus
///             the number of rows of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param[in]  num_rows
///             Number of rows of the matrix \c A. This is essentially the size
///             of \c A_index_pointer array minus one.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

static void csr_matvec(
		const DataType* A_data,
		const LongIndexType* A_column_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const LongIndexType num_rows,
		DataType* c)
{
	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;
	long double sum;

	for (row=0; row < num_rows; ++row)
	{
		sum = 0.0;
		for (index_pointer=A_index_pointer[row];
			 index_pointer < A_index_pointer[row+1];
			 ++index_pointer)
		{
			column = A_column_indices[index_pointer];
			sum += A_data[index_pointer] * b[column];
		}
		c[row] = static_cast<DataType>(sum);
	}
}


// ===============
// csr matvec plus
// ===============

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is
///                compressed sparse row (CSR) matrix and \f$ \boldsymbol{b}
///                \f$ is a dense vector. The output \f$ \boldsymbol{c} \f$ is
///                a dense vector.
///
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
///
/// \param[in]     A_data
///                CSR format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_column_indices
///                CSR format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSR format index pointer. The length of this array is one
///                plus the number of rows of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of the matrix \c A. This is essentially the
///                size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

static void csr_matvec_plus(
		const DataType* A_data,
		const LongIndexType* A_column_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const DataType alpha,
		const LongIndexType num_rows,
		DataType* c)
{
	if (alpha == 0.0)
	{
		return;
	}

	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;
	long double sum;

	for (row=0; row < num_rows; ++row)
	{
		sum = 0.0;
		for (index_pointer=A_index_pointer[row];
			 index_pointer < A_index_pointer[row+1];
			 ++index_pointer)
		{
			column = A_column_indices[index_pointer];
			sum += A_data[index_pointer] * b[column];
		}
		c[row] += alpha * static_cast<DataType>(sum);
	}
}


// =====================
// csr transposed matvec
// =====================

/// \brief      Computes \f$\boldsymbol{c} =\mathbf{A}^{\intercal}
///             \boldsymbol{b}\f$ where \f$ \mathbf{A} \f$ is compressed sparse
///             row (CSR) matrix and \f$ \boldsymbol{b} \f$ is a dense vector.
///             The output \f$ \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]  A_data
///             CSR format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_column_indices
///             CSR format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSR format index pointer. The length of this array is one plus
///             the number of rows of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param[in]  num_rows
///             Number of rows of the matrix \c A. This is essentially the size
///             of \c A_index_pointer array minus one.
/// \param[in]  num_columns
///             Number of columns of the matrix \c A.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

static void csr_transposed_matvec(
		const DataType* A_data,
		const LongIndexType* A_column_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const LongIndexType num_rows,
		const LongIndexType num_columns,
		DataType* c)
{
	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;

	// Initialize output to zero
	for (column=0; column < num_columns; ++column)
	{
		c[column] = 0.0;
	}

	for (row=0; row < num_rows; ++row)
	{
		for (index_pointer=A_index_pointer[row];
			 index_pointer < A_index_pointer[row+1];
			 ++index_pointer)
		{
			column = A_column_indices[index_pointer];
			c[column] += A_data[index_pointer] * b[row];
		}
	}
}


// ==========================
// csr transposed matvec plus
// ==========================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b}\f$ where \f$
///                \mathbf{A} \f$ is compressed sparse row (CSR) matrix and \f$
///                \boldsymbol{b} \f$ is a dense vector. The output \f$
///                \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]     A_data
///                CSR format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_column_indices
///                CSR format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSR format index pointer. The length of this array is one
///                plus the number of rows of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of the matrix \c A. This is essentially the
///                size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

static void csr_transposed_matvec_plus(
		const DataType* A_data,
		const LongIndexType* A_column_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const DataType alpha,
		const LongIndexType num_rows,
		DataType* c)
{
	if (alpha == 0.0)
	{
		return;
	}

	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;

	for (row=0; row < num_rows; ++row)
	{
		for (index_pointer=A_index_pointer[row];
			 index_pointer < A_index_pointer[row+1];
			 ++index_pointer)
		{
			column = A_column_indices[index_pointer];
			c[column] += alpha * A_data[index_pointer] * b[row];
		}
	}
}


// ==========
// csc matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A} \f$ is compressed sparse column (CSC)
///             matrix and \f$ \boldsymbol{b} \f$ is a dense vector. The output
///             \f$ \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]  A_data
///             CSC format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_row_indices
///             CSC format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSC format index pointer. The length of this array is one plus
///             the number of columns of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param[in]  num_rows
///             Number of rows of the matrix \c A.
/// \param[in]  num_columns
///             Number of columns of the matrix \c A. This is essentially the
///             size of \c A_index_pointer array minus one.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

static void csc_matvec(
		const DataType* A_data,
		const LongIndexType* A_row_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const LongIndexType num_rows,
		const LongIndexType num_columns,
		DataType* c)
{
	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;

	// Initialize output to zero
	for (row=0; row < num_rows; ++row)
	{
		c[row] = 0.0;
	}

	for (column=0; column < num_columns; ++column)
	{
		for (index_pointer=A_index_pointer[column];
			 index_pointer < A_index_pointer[column+1];
			 ++index_pointer)
		{
			row = A_row_indices[index_pointer];
			c[row] += A_data[index_pointer] * b[column];
		}
	}
}


// ===============
// csc matvec plus
// ===============

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is
///                compressed sparse column (CSC) matrix and \f$ \boldsymbol{b}
///                \f$ is a dense vector. The output \f$ \boldsymbol{c} \f$ is
///                a dense vector.
///
/// \param[in]     A_data
///                CSC format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_row_indices
///                CSC format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSC format index pointer. The length of this array is one
///                plus the number of columns of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_columns
///                Number of columns of the matrix \c A. This is essentially
///                the size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

static void csc_matvec_plus(
		const DataType* A_data,
		const LongIndexType* A_row_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const DataType alpha,
		const LongIndexType num_columns,
		DataType* c)
{
	if (alpha == 0.0)
	{
		return;
	}

	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;

	for (column=0; column < num_columns; ++column)
	{
		for (index_pointer=A_index_pointer[column];
			 index_pointer < A_index_pointer[column+1];
			 ++index_pointer)
		{
			row = A_row_indices[index_pointer];
			c[row] += alpha * A_data[index_pointer] * b[column];
		}
	}
}


// =====================
// csc transposed matvec
// =====================

/// \brief      Computes \f$\boldsymbol{c} =\mathbf{A}^{\intercal}
///             \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is compressed
///             sparse column (CSC) matrix and \f$ \boldsymbol{b} \f$ is a
///             dense vector. The output \f$ \boldsymbol{c} \f$ is a dense
///             vector.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c
///             float, the summation loses the precision, especially when the
///             vector size is large. It seems that using <tt>long double</tt>
///             is slightly faster than using \c double. The advantage of using
///             a type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
///
/// \param[in]  A_data
///             CSC format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_row_indices
///             CSC format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSC format index pointer. The length of this array is one plus
///             the number of columns of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param      num_columns
///             Number of columns of the matrix \c A. This is essentially the
///             size of \c A_index_pointer array minus one.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

static void csc_transposed_matvec(
		const DataType* A_data,
		const LongIndexType* A_row_indices,
		const LongIndexType* A_index_pointer,
		const DataType* b,
		const LongIndexType num_columns,
		DataType* c)
{
	LongIndexType index_pointer;
	LongIndexType row;
	LongIndexType column;
	long double sum;

	for (column=0; column < num_columns; ++column)
	{
		sum = 0.0;
		for (index_pointer=A_index_pointer[column];
			 index_pointer < A_index_pointer[column+1];
			 ++index_pointer)
		{
			row = A_row_indices[index_pointer];
			sum += A_data[index_pointer] * b[row];
		}
		c[column] = static_cast<DataType>(sum);
	}
}


// ==========================
// csc transposed matvec plus
// ==========================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$
///                \mathbf{A} \f$ is compressed sparse column (CSC) matrix and
///                \f$ \boldsymbol{b} \f$ is a dense vector. The output \f$
///                \boldsymbol{c} \f$ is a dense vector.
///
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
///
/// \param[in]     A_data
///                CSC format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_row_indices
///                CSC format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSC format index pointer. The length of this array is one
///                plus the number of columns of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param         num_columns
///                Number of columns of the matrix \c A. This is essentially
///                the size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

	static void csc_transposed_matvec_plus(
			const DataType* A_data,
			const LongIndexType* A_row_indices,
			const LongIndexType* A_index_pointer,
			const DataType* b,
			const DataType alpha,
			const LongIndexType num_columns,
			DataType* c)
	{
		if (alpha == 0.0)
		{
			return;
		}

		LongIndexType index_pointer;
		LongIndexType row;
		LongIndexType column;
		long double sum;

		for (column=0; column < num_columns; ++column)
		{
			sum = 0.0;
			for (index_pointer=A_index_pointer[column];
				index_pointer < A_index_pointer[column+1];
				++index_pointer)
			{
				row = A_row_indices[index_pointer];
				sum += A_data[index_pointer] * b[row];
			}
			c[column] += static_cast<DataType>(alpha * sum);
		}
	}


// ==================
// create band matrix
// ==================

/// \brief      Creates bi-diagonal or symmetric tri-diagonal matrix from the
///             diagonal array (\c diagonals) and off-diagonal array (\c
///             supdiagonals).
///
/// \details    The output is written in place (in \c matrix). The output is
///             only written up to the \c non_zero_size element, that is: \c
///             matrix[:non_zero_size,:non_zero_size] is filled, and the rest
///             is assumed to be zero.
///
///             Depending on \c tridiagonal, the matrix is upper bi-diagonal or
///             symmetric tri-diagonal.
///
/// \param[in]  diagonals
///             An array of length \c n. All elements \c diagonals create the
///             diagonals of \c matrix.
/// \param[in]  supdiagonals
///             An array of length \c n. Elements \c supdiagonals[0:-1] create
///             the upper off-diagonal of \c matrix, making \c matrix an upper
///             bi-diagonal matrix. In addition, if \c tridiagonal is set to
///             \c 1, the lower off-diagonal is also created similar to the
///             upper off-diagonal, making \c matrix a symmetric tri-diagonal
///             matrix.
/// \param[in]  non_zero_size
///             Up to the \c matrix[:non_zero_size,:non_zero_size] of \c matrix
///             will be written. At most, \c non_zero_size can be \c n, which
///             is the size of \c diagonals array and the size of the square
///             matrix. If \c non_zero_size is less than \c n, it is due to the
///             fact that either \c diagonals or \c supdiagonals has zero
///             elements after the \c size element (possibly due to early
///             termination of Lanczos iterations method).
/// \param[in]  tridiagonal
///             Boolean. If set to \c 0, the matrix \c T becomes upper
///             bi-diagonal. If set to \c 1, the matrix becomes symmetric
///             tri-diagonal.
/// \param[out] matrix
///             A 2D  matrix (written in place) of the shape \c (n,n). This is
///             the output of this function. This matrix is assumed to be
///             initialized to zero before calling this function.
	static void create_band_matrix(
			const DataType* diagonals,
			const DataType* supdiagonals,
			const IndexType non_zero_size,
			const FlagType tridiagonal,
			DataType** matrix)
	{
		for (IndexType j=0; j < non_zero_size; ++j)
		{
			// Diagonals
			matrix[j][j] = diagonals[j];

			// Off diagonals
			if (j < non_zero_size-1)
			{
				// Sup-diagonal
				matrix[j][j+1] = supdiagonals[j];

				// Sub-diagonal, making symmetric tri-diagonal matrix
				if (tridiagonal)
				{
					matrix[j+1][j] = supdiagonals[j];
				}
			}
		}
	}

};

// ===============================
// Explicit static template instantiation
// ===============================

template struct cMatrixOperations<float>;
template struct cMatrixOperations<double>;
template struct cMatrixOperations<long double>;

#endif  // _C_BASIC_ALGEBRA_C_MATRIX_OPERATIONS_H_

