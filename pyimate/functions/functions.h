/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_FUNCTIONS_H_
#define FUNCTIONS_FUNCTIONS_H_

// ========
// Function
// ========

/// \brief   Defines the function \f$ f: \lambda \mapsto \lambda \f$.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.
///
/// \note    This class is a base class for other classes and serves as an
///          interface. To create a new matrix function, derive a class from
///          \c Function class and implement the \c function method.

class Function
{
    public:
        virtual ~Function();
        virtual float function(const float lambda_) const = 0;
        virtual double function(const double lambda_) const = 0;
        virtual long double function(const long double lambda_) const = 0;
};


// ========
// Identity
// ========

/// \brief   Defines the function \f$ f: \lambda \mapsto \lambda \f$.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Identity : public Function
{
    public:
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
};


// =======
// Inverse
// =======

/// \brief   Defines the function \f$ f: \lambda \mapsto \frac{1}{\lambda} \f$.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Inverse : public Function
{
    public:
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
};


// =========
// Logarithm
// =========

/// \brief   Defines the function \f$ f: \lambda \mapsto \log(\lambda) \f$.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Logarithm : public Function
{
    public:
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
};


// ===========
// Exponential
// ===========

/// \brief   Defines the function \f$ f: \lambda \mapsto e^{\lambda} \f$.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Exponential : public Function
{
    public:
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
};


// =====
// Power
// =====

/// \brief   Defines the function \f$ f: \lambda \mapsto \lambda^{p} \f$,
///          where \f$ p \in \mathbb{R} \f$ is a parameter and should be set by
///          \c this->exponent member.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Power : public Function
{
    public:
        Power();
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double exponent;
};


// ========
// Gaussian
// ========

/// \brief   Defines the function
///          \f[
///              f: \lambda \mapsto \frac{1}{\sigma \sqrt{2 \pi}}
///              e^{-\frac{1}{2} \frac{(\lambda - \mu)^2}{\sigma^2}},
///          \f]
///          where \f$ \mu \f$ and \f$ \sigma \f$ parameters are the mean and
///          standard deviation of the Gaussian function and should be set by
///          \c this->mu and \c this->sigma members, respectively.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Gaussian : public Function
{
    public:
        Gaussian();
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double mu;
        double sigma;
};


// ===========
// Smooth Step
// ===========

/// \brief   Defines the function
///          \f[
///              f: \lambda \mapsto \frac{1}{2}
///              \left( 1 + \mathrm{tanh}(\alpha \lambda) \right)
///          \f]
///          where \f$ \alpha \f$ is a scale parameter and should be set by
///          \c this->alpha member.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$
///          is used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.
///
/// \note    The smooth step function defined here should not be confused with
///          a conventionally used function of the same name using cubic
///          polynomial.

class SmoothStep : public Function
{
    public:
        SmoothStep();
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double alpha;
};



#endif  // FUNCTIONS_FUNCTIONS_H_
