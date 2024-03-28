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

template< std::floating_point F > 
auto deflate_transform(const DenseMatrix< F >& Q) -> std::function< void(F*, const size_t)>{
  const auto deflate = [Q](F* x, const size_t n){
    Eigen::Map< Vector< F > > x_vec(x, n);
    Vector< F > x_copy = x_vec; // save copy
    x_vec = x_vec - Q * (Q.adjoint() * x_copy);
  };
  return deflate;
}

template< std::floating_point F, class Matrix, LinearOperator Wrapper >
// requires std::invocable< Wrapper, const Matrix* >
void _matrix_function_wrapper(py::module& m, std::string prefix){
  prefix += std::string("_MatrixFunction");
  const char* cls_name = prefix.c_str();
  
  // using Wrapped_t = decltype(wrap(static_cast< const Matrix* >(nullptr)));
  using OP_t = MatrixFunction< F, Wrapper >;
  py::class_< OP_t >(m, cls_name)
    .def(py::init([](const Matrix& A, const int deg, const F rtol, const int orth, const int ncv, const py::kwargs& kwargs) {
      // const auto sf = param_spectral_func< F >(kwargs);
      const Wrapper* op = new Wrapper(A); // todo: add custom destructor to delete this
      bool is_native = true; 
      const auto sf = param_vector_func< F >(kwargs, is_native);
      // const auto op = Wrapper(*A); // we want a new instance here!
      // std::cout << op.shape().first << ", ";
      return std::make_unique< OP_t >(OP_t(*op, sf, deg, rtol, orth, ncv, is_native));
    }))
    .def_property_readonly("shape", &OP_t::shape)
    .def_property_readonly("dtype", [](const OP_t& M) -> py::dtype {
      auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
      return dtype; 
    })
    .def_property("fun", [](const OP_t& M){
      return py::cpp_function([&M](py::array_t< F >& x){
        M.f(x.mutable_data(), x.size());
        return x; 
      });
    }, [](OP_t& M, const py::object fun, py::kwargs& kwargs){
      if (py::isinstance< py::str >(fun)) {
        kwargs["function"] = fun; 
        M.f = param_vector_func< F >(kwargs, M.native_f);
      } else {
        // Try to deduce scalar-valued input/output vs vector-valued
        const py::function g = fun.cast< const py::function >();
        auto ov = deduce_vector_func< F >(g, M.native_f);
        if (ov){ M.f = *ov; return; }
        throw std::invalid_argument("Invalid function type; matrix function must be vector-valued.");
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
    .def_property_readonly("native_fun", [](const OP_t& M) -> bool { return M.native_f; })
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
    .def_property_readonly("transform", [](const OP_t& M){ 
      // return py::cpp_function(M.transform); 
      return py::cpp_function([&M](py::array_t< F >& x){
        M.transform(x.mutable_data(), x.size());
        return x; 
      });
    })
    // .def("_set_transform", [](OP_t& M, const py::object fun, py::kwargs& kwargs){
    //   if (py::isinstance< py::str >(fun)) {
    //     // kwargs["Q"] = fun; 
    //     M.transform = param_vector_func< F >(kwargs, M.native_f);
    //   } else if (py::isinstance< py::function >(fun)){
    //     // See also: https://github.com/pybind/pybind11/blob/master/tests/test_callbacks.cpp
    //     const py::function g = fun.cast< const py::function >();
    //     auto of = deduce_vector_func< F >(g, M.native_f);
    //     if (of){
    //       M.transform = std::move(*of);
    //     }
    //     throw std::invalid_argument("Failed to deduce transform function.");
    //   } else {
    //     throw std::invalid_argument("Invalid transform function specified. Must be Callable or string.");
    //   }
    // })
    .def("deflate", [](OP_t& M, const DenseMatrix< F >& Q){
      M.transform = deflate_transform(Q);
    })
    .def("reset_transform", [](OP_t& M, const DenseMatrix< F >& Q){
      M.transform = [](F* v, const size_t N){ return; };
    })
    .def("quad_sum", [](const OP_t& M, const DenseMatrix< F >& W) -> py::tuple {
      F quad_sum = 0.0; 
      const size_t N = static_cast< size_t >(W.cols());
      auto estimates = static_cast< Array< F > >(Array< F >::Zero(N));
      auto y = static_cast< Vector< F > >(Vector< F >::Zero(W.rows()));

      // TODO: make parallel
      for (size_t j = 0; j < N; ++j){
        M.matvec(W.col(j).data(), y.data());
        estimates[j] = W.col(j).adjoint().dot(y);
        quad_sum += estimates[j];
      }
      return py::make_tuple(quad_sum, py::cast(estimates));
    })
    ; 
}

void native_exp(float* x, const size_t n) { 
  std::for_each_n(x, n, [](auto& ew){ ew = std::exp(ew); });
}

PYBIND11_MODULE(_operators, m) {
  m.doc() = "operators module";
  _matrix_function_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m, "DenseF");
  _matrix_function_wrapper< double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m, "DenseD");

  _matrix_function_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float, false > >(m, "SparseF");
  _matrix_function_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double, false > >(m, "SparseD");
  
  _matrix_function_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float, true > >(m, "SparseFG");
  _matrix_function_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double, true > >(m, "SparseDG");
  
  _matrix_function_wrapper< float, py::object, PyLinearOperator< float > >(m, "GenericF");
  _matrix_function_wrapper< double, py::object, PyLinearOperator< double > >(m, "GenericD");

  m.def("exp", &native_exp);
};

// Extra to not lose: 
// template< std::floating_point F, class Matrix, typename WrapperFunc >
// requires std::invocable< WrapperFunc, const Matrix* >
// using WrapperType = decltype(wrap(static_cast< const Matrix* >(nullptr)));