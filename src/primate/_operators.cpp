#include <type_traits> // result_of
#include <cmath> // constants
// #include <iostream>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "_operators/linear_operator.h"
#include "_random_generator/vector_generator.h"
#include "_lanczos/lanczos.h"
#include "eigen_core.h"
#include "spectral_functions.h"
#include "eigen_operators.h"
#include "pylinop.h"

namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;


template< std::floating_point F, LinearOperator Matrix >
auto py_matvec(const Matrix& A, const py_array< F >& x) -> py_array< F > {
  using ArrayF = Array< F >;
  using VectorF = Vector< F >;
  if (size_t(A.shape().second) != size_t(x.size())){
    throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator.");
  }
  auto output = static_cast< ArrayF >(VectorF::Zero(A.shape().first));
  A.matvec(x.data(), output.data());
  return py::cast(output);
}

template< std::floating_point F, LinearOperator Matrix >
void py_matvec_inplace(const Matrix& A, const py_array< F >& x, py_array< F >& y) {
  if (size_t(A.shape().second) != size_t(x.size()) || size_t(A.shape().first) != size_t(y.size())){
    throw std::invalid_argument("Input/output dimension mismatch; vector inputs must match shape of the operator.");
  }
  A.matvec(x.data(), y.mutable_data());
}

template< std::floating_point F, LinearOperator Matrix > 
auto py_matmat(const Matrix& M, const py_array< F >& X) -> py_array< F >{
  if constexpr (SupportsMatrixMult< Matrix >){
    if (size_t(X.ndim() == 1)){
      if (size_t(M.shape().second) != size_t(X.size())){ throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator."); }
      auto Y = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(M.shape().second, 1));
      M.matmat(X.data(), Y.data(), 1);
      return py::cast(Y);
    } else if (X.ndim() == 2){
      if (size_t(M.shape().second) != size_t(X.shape(0))){ throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator."); }
      const auto k = X.shape(1);
      auto Y = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(M.shape().second, k));
      M.matmat(X.data(), Y.data(), k);
      return py::cast(Y);
    } else {
      throw std::invalid_argument("Input dimension mismatch; input must be 1 or 2-dimensional.");
    }
  } else {
    return X;
  }
}

template< std::floating_point F, LinearOperator Matrix >
auto py_quad(const Matrix& A, const py_array< F >& x) -> F {
  if (size_t(A.shape().second) != size_t(x.size())){
    throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator.");
  }
  return A.quad(x.data());
}


// template< std::floating_point F, typename WrapperFunc >
// requires std::invocable< WrapperFunc, const Matrix* >
// template< std::floating_point F, class Op, LinearOperator Matrix >
// void _operators_wrapper(py::module& m, std::string prefix){
//   // using WrapperType = decltype(wrap(static_cast< const Matrix* >(nullptr)));
//   prefix += std::string("Matrix");
//   const char* cls_name = prefix.c_str();
//   py::class_< Matrix >(m, cls_name)
//     .def(py::init< const Op >())
//     .def_property_readonly("shape", &Matrix::shape)
//     .def_property_readonly("dtype", [](const Matrix& M) {
//       return pybind11::dtype(pybind11::format_descriptor< F >::format());
//     })
//     .def("matvec", &py_matvec< F, Matrix >)
//     .def("matvec", &py_matvec_inplace< F, Matrix >)
//     .def("matmat", &py_matmat< F, Matrix >)
//     .def("__matmul__", &py_matmat< F, Matrix >) // see: https://peps.python.org/pep-0465/
//     ;
// }

// Template function for generating module definitions for a given Operator / precision 
// template< std::floating_point F, LinearOperator Matrix >
// void _matrix_function_wrapper(py::module& m, std::string prefix){
//   prefix += std::string("_MatrixFunction");
//   const char* cls_name = prefix.c_str();
//   using OP_t = MatrixFunction< F, Matrix >;
//   py::class_< OP_t >(m, cls_name)
//     .def(py::init([](const Matrix* A, const int deg, const F rtol, const int orth, const int ncv, const py::kwargs& kwargs) {
//       const auto sf = param_spectral_func< F >(kwargs);
//       return std::unique_ptr< OP_t >(new OP_t(A, sf, deg, rtol, orth, ncv));
//     }))
//     .def_property_readonly("shape", &OP_t::shape)
//     .def_property_readonly("dtype", [](const OP_t& M) -> py::dtype {
//       auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
//       return dtype; 
//     })
//     .def_readonly("deg", &OP_t::deg)
//     .def_readwrite("rtol", &OP_t::rtol)
//     .def_readwrite("orth", &OP_t::orth)
//     .def("matvec", &py_matvec< F, OP_t >)
//     .def("matvec", &py_matvec_inplace< F, OP_t >)
//     .def("matmat", &py_matmat< F, OP_t >)
//     .def("__matmul__", &py_matmat< F, OP_t >)
//     ; 
// }


template< std::floating_point F, class Matrix, LinearOperator Wrapper >
// requires std::invocable< Wrapper, const Matrix* >
void _matrix_function_wrapper(py::module& m, std::string prefix){
  prefix += std::string("_MatrixFunction");
  const char* cls_name = prefix.c_str();
  
  // using Wrapped_t = decltype(wrap(static_cast< const Matrix* >(nullptr)));
  using OP_t = MatrixFunction< F, Wrapper >;
  py::class_< OP_t >(m, cls_name)
    .def(py::init([](const Matrix& A, const int deg, const F rtol, const int orth, const int ncv, const py::kwargs& kwargs) {
      const auto sf = param_spectral_func< F >(kwargs);
      // const auto op = Wrapper(*A); // we want a new instance here!
      // std::cout << op.shape().first << ", ";
      const Wrapper* op = new Wrapper(A); // todo: add custom destructor to delete this
      return std::make_unique< OP_t >(OP_t(*op, sf, deg, rtol, orth, ncv));
    }))
    .def_property_readonly("shape", &OP_t::shape)
    .def_property_readonly("dtype", [](const OP_t& M) -> py::dtype {
      auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
      return dtype; 
    })
    .def_property("fun", [](const OP_t& M){
      return py::cpp_function(M.f);
    }, [](OP_t& M, const py::object fun, py::kwargs& kwargs){
      if (py::isinstance< py::str >(fun)) {
        kwargs["function"] = fun; 
        M.f = param_spectral_func< F >(kwargs);
      } else {
        // See also: https://github.com/pybind/pybind11/blob/master/tests/test_callbacks.cpp
        std::function< F(F) > f = py::cast< std::function< F(F) > >(fun);
        using fn_type = F(*)(F);
        const auto *result = f.template target< fn_type >();
        if (!result) {
          // std::cout << "Failed to convert to function ptr! Falling back to pyfunc" << std::endl;
          M.f = [fun](F x) -> F { return fun(x).template cast< F >(); };
        } else {
          // std::cout << "Native cpp function detected!" << std::endl;
          M.f = f; 
        }
        // if (py::isinstance< py::function >(fun)){
        // std::cout << "Python function detected!" << std::endl;
        // M.f = [fun](F x) -> F { return fun(x).template cast< F >(); };
        // } else {
        //   throw std::invalid_argument("Invalid argument type; must be string or Callable");
        // }
      }
    })
    .def_readonly("deg", &OP_t::deg)
    .def_readonly("ncv", &OP_t::ncv)
    .def_readwrite("rtol", &OP_t::rtol)
    .def_readwrite("orth", &OP_t::orth)
    .def("matvec", &py_matvec< F, OP_t >)
    .def("matvec", &py_matvec_inplace< F, OP_t >)
    .def("matmat", &py_matmat< F, OP_t >)
    .def("__matmul__", &py_matmat< F, OP_t >)
    .def("quad", &py_quad< F, OP_t >)
    .def_property_readonly("nodes", [](const OP_t& M){ return py::cast(M.nodes); })
    .def_property_readonly("weights", [](const OP_t& M){ return py::cast(M.weights); })
    .def_property_readonly("_alpha", [](const OP_t& M){ return py::cast(M.alpha); })
    .def_property_readonly("_beta", [](const OP_t& M){ return py::cast(M.beta); })
    .def_property_readonly("krylov_basis", [](const OP_t& M){ return py::cast(M.Q); })
    .def_property("method", [](const OP_t& M){
      return M.wgt_method == golub_welsch ? "golub_welsch" : "fttr";
    }, [](OP_t& M, std::string method){
      if (method == "golub_welsch"){
        M.wgt_method = golub_welsch; 
      } else if (method == "fttr"){
        M.wgt_method = fttr;
      } else {
        throw std::invalid_argument("Invalid method supplied. Must be one of 'golub_welsch' or 'fttr'.");
      }
    })
    .def_property("transform", [](const OP_t& M){
        return py::cpp_function(M.f);
      }, [](OP_t& M, const py::object fun, py::kwargs& kwargs){
        if (py::isinstance< py::str >(fun)) {
          // kwargs["Q"] = fun; 
          M.transform = param_vector_func< F >(M.shape().second, kwargs);
        } else {
          // See also: https://github.com/pybind/pybind11/blob/master/tests/test_callbacks.cpp
          std::function< void(F*) > f = py::cast< std::function< void(F*) > >(fun);
          using fn_type = void(*)(F*);
          const auto *result = f.template target< fn_type >();
          if (!result) {
            // std::cout << "Failed to convert to function ptr! Falling back to pyfunc" << std::endl;
            M.transform = [fun](F* x) -> void { fun(x); return; };
          } else {
            // std::cout << "Native cpp function detected!" << std::endl;
            M.transform = f; 
          }
        }
      }
    )
    ; 
}

float native_exp(float x) { return std::exp(x); }

PYBIND11_MODULE(_operators, m) {
  m.doc() = "operators module";
  _matrix_function_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m, "DenseF");
  _matrix_function_wrapper< double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m, "DenseD");

  _matrix_function_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float > >(m, "SparseF");
  _matrix_function_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double > >(m, "SparseD");
  
  _matrix_function_wrapper< float, py::object, PyLinearOperator< float > >(m, "GenericF");
  _matrix_function_wrapper< double, py::object, PyLinearOperator< double > >(m, "GenericD");

  m.def("exp", &native_exp);
};

// Extra to not lose: 
// template< std::floating_point F, class Matrix, typename WrapperFunc >
// requires std::invocable< WrapperFunc, const Matrix* >
// using WrapperType = decltype(wrap(static_cast< const Matrix* >(nullptr)));