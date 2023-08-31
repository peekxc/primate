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

// Before including cmath, define _USE_MATH_DEFINES. This is only required to
// define the math constants like M_PI, etc, in win32 operating system.
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && \
    !defined(__CYGWIN__)
    #define _USE_MATH_DEFINES
#endif

#include "./functions.h"
#include <cmath>  // log, exp, pow, tanh, M_SQRT1_2, M_2_SQRTPI, NAN


// ===================
// Function destructor
// ===================

/// \brief Default virtual destructor.
///

Function::~Function()
{
}


// =================
// Identity function (float)
// =================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Identity::function(const float lambda_) const
{
    return lambda_;
}


// =================
// Identity function (double)
// =================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Identity::function(const double lambda_) const
{
    return lambda_;
}


// =================
// Identity function (long double)
// =================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Identity::function(const long double lambda_) const
{
    return lambda_;
}


// ================
// Inverse function (float)
// ================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Inverse::function(const float lambda_) const
{
    return 1.0 / lambda_;
}


// ================
// Inverse function (double)
// ================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Inverse::function(const double lambda_) const
{
    return 1.0 / lambda_;
}


// ================
// Inverse function (long double)
// ================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Inverse::function(const long double lambda_) const
{
    return 1.0 / lambda_;
}


// ==================
// Logarithm function (float)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Logarithm::function(const float lambda_) const
{
    return log(lambda_);
}


// ==================
// Logarithm function (double)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Logarithm::function(const double lambda_) const
{
    return log(lambda_);
}


// ==================
// Logarithm function (long double)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Logarithm::function(const long double lambda_) const
{
    return log(lambda_);
}


// ====================
// Exponential function (float)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Exponential::function(const float lambda_) const
{
    return exp(lambda_);
}


// ====================
// Exponential function (double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Exponential::function(const double lambda_) const
{
    return exp(lambda_);
}


// ====================
// Exponential function (long double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Exponential::function(const long double lambda_) const
{
    return exp(lambda_);
}


// =====
// Power
// =====

/// \brief Sets the default for the parameter \c exponent to \c 2.0.
///

Power::Power():
    exponent(2.0)
{
}


// ==============
// Power function (float)
// ==============

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Power::function(const float lambda_) const
{
    return pow(lambda_, static_cast<float>(this->exponent));
}


// ==============
// Power function (double)
// ==============

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Power::function(const double lambda_) const
{
    return pow(lambda_, this->exponent);
}


// ==============
// Power function (long double)
// ==============

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Power::function(const long double lambda_) const
{
    return pow(lambda_, static_cast<long double>(this->exponent));
}


// ========
// Gaussian
// ========

/// \brief Sets the default for the parameter \c mu to \c 0.0 and for the
///        parameter \c sigma to \c 1.0.

Gaussian::Gaussian():
    mu(0.0), sigma(1.0)
{
}

// =================
// Gaussian function (float)
// =================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Gaussian::function(const float lambda_) const
{
    float mu_ = static_cast<float>(this->mu);
    float sigma_ = static_cast<float>(this->sigma);
    float x = (lambda_ - mu_) / sigma_;
    return (0.5 * M_SQRT1_2 * M_2_SQRTPI / sigma_) * exp(-0.5 * x * x);
}


// =================
// Gaussian function (double)
// =================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Gaussian::function(const double lambda_) const
{
    double x = (lambda_ - this->mu) / this->sigma;
    return (0.5 * M_SQRT1_2 * M_2_SQRTPI / this->sigma) * exp(-0.5 * x * x);
}


// =================
// Gaussian function (long double)
// =================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Gaussian::function(const long double lambda_) const
{
    long double mu_ = static_cast<long double>(this->mu);
    long double sigma_ = static_cast<long double>(this->sigma);
    long double x = (lambda_ - mu_) / sigma_;
    return (0.5 * M_SQRT1_2 * M_2_SQRTPI / sigma_) * exp(-0.5 * x * x);
}


// ===========
// Smooth Step
// ===========

/// \brief Sets the default for the parameter \c alpha to \c 1.0.
///

SmoothStep::SmoothStep():
    alpha(1.0)
{
}


// ====================
// Smooth Step function (float)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float SmoothStep::function(const float lambda_) const
{
    return 0.5 * (1.0 + tanh(static_cast<float>(this->alpha) * lambda_));
}


// ====================
// Smooth Step function (double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double SmoothStep::function(const double lambda_) const
{
    return 0.5 * (1.0 + tanh(this->alpha * lambda_));
}


// ====================
// Smooth Step function (long double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double SmoothStep::function(const long double lambda_) const
{
    return 0.5 * (1.0 + tanh(static_cast<long double>(this->alpha) * lambda_));
}
