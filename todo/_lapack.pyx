# cython: language_level=3, boundscheck=False, language='c++'
# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from scipy.linalg.cython_lapack cimport sstev, dstev, sbdsdc, dbdsdc     # noqa


# ============
# lapack sstev
# ============

cdef public void lapack_sstev(char* jobz, int* n, float* d, float* e, float* z,
                              int* ldz, float* work, int* info) nogil:
    """
    Wrapper for cython's lapack's ``sstev`` function. This function is defined
    as ``public``, so that cython generates a C++ header file that can be
    included in  c++ code. See ``lapack_api.h``.

    .. note::

        To generate cython's api, this file should be included in the cython's
        pyx module. See ``py_c_trace_estimator.pyx``.
    """

    sstev(jobz, n, d, e, z, ldz, work, info)


# ============
# lapack dstev
# ============

cdef public void lapack_dstev(char* jobz, int* n, double* d, double* e,
                              double* z, int* ldz, double* work,
                              int* info) nogil:
    """
    Wrapper for cython's lapack's ``dstev`` function. This function is defined
    as ``public``, so that cython generates a C++ header file that can be
    included in  c++ code. See ``lapack_api.h``.

    .. note::

        To generate cython's api, this file should be included in the cython's
        pyx module. See ``py_c_trace_estimator.pyx``.
    """

    dstev(jobz, n, d, e, z, ldz, work, info)


# =============
# lapack sbdsdc
# =============

cdef public void lapack_sbdsdc(char* uplo, char* compq, int* n, float* d,
                               float *e, float* u, int* ldu, float* vt,
                               int* ldvt, float* q, int* iq, float* work,
                               int* iwork, int* info) nogil:
    """
    Wrapper for cython's lapack's ``sbdsdc`` function. This function is defined
    as ``public``, so that cython generates a C++ header file that can be
    included in  c++ code. See ``lapack_api.h``.

    .. note::

        To generate cython's api, this file should be included in the cython's
        pyx module. See ``py_c_trace_estimator.pyx``.
    """

    sbdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork, info)


# =============
# lapack dbdsdc
# =============

cdef public void lapack_dbdsdc(char* uplo, char* compq, int* n, double* d,
                               double *e, double* u, int* ldu, double* vt,
                               int* ldvt, double* q, int* iq, double* work,
                               int* iwork, int* info) nogil:
    """
    Wrapper for cython's lapack's ``dbdsdc`` function. This function is defined
    as ``public``, so that cython generates a C++ header file that can be
    included in  c++ code. See ``lapack_api.h``.

    .. note::

        To generate cython's api, this file should be included in the cython's
        pyx module. See ``py_c_trace_estimator.pyx``.
    """

    dbdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork, info)