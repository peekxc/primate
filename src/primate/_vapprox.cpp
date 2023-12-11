#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "_random_generator/vector_generator.h"
#include "_lanczos/lanczos.h"
#include "eigen_operators.h"
#include <cmath> // constants
#include <iostream>
#include <stdio.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


// template< std::floating_point F, LinearOperator Matrix > 
// struct MatrixFunction {
//   using VectorF = Eigen::Matrix< F, Dynamic, 1 >;
//   using ArrayF = Eigen::Array< F, Dynamic, 1 >;
//   using EigenSolver = Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >; 

//   // Parameters 
//   const Matrix& op; 
//   std::function< F(F) > f; 
//   const int deg;
//   const F rtol; 
//   const int orth;

//   MatrixFunction(const Matrix& A, std::function< F(F) > fun, int lanczos_degree, F lanczos_rtol, int add_orth) 

// PYBIND11_MODULE(_vapprox, m) {
//   _matrix_function_wrapper< float, Eigen::SparseMatrix< float > >(m); 
// }

