#ifndef _PYLINOPS_H
#define _PYLINOPS_H

#include <type_traits>
#include <concepts>
#include <algorithm>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <_definitions/definitions.h>
#include <_definitions/types.h>
#include <_diagonalization/diagonalization.h>
#include <_diagonalization/lanczos_tridiagonalization.h>
#include <_diagonalization/golub_kahn_bidiagonalization.h>
#include <_linear_operator/linear_operator.h>
#include <_c_basic_algebra/c_matrix_operations.h>

#include "pylinops.h"
namespace py = pybind11;

using py_arr_f = py::array_t< float, py::array::c_style | py::array::forcecast >;

namespace py = pybind11;

template< typename F = float > 
struct PyDiagonalOperator {
  vector< F > _diagonals; 
  
  PyDiagonalOperator(const py::array_t< F > diagonals) {
    py::buffer_info buffer = diagonals.request();
    const F* data = static_cast< const F* >(buffer.ptr);
    _diagonals = std::vector(data, data + buffer.size);
  }
  
  // Needed to match the LinearOperator concept
  auto matvec(const F* inp, F* out) const -> void {
    std::transform(inp, inp + _diagonals.size(), _diagonals.begin(), out, std::multiplies< F >());
  }
  auto matvec(const py::array_t< F >& input) const -> py::array_t< F > {
    auto out = vector< F >(_diagonals.size());
    this->matvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  auto shape() const -> pair< size_t, size_t > { 
    return std::make_pair(_diagonals.size(), _diagonals.size());
  }

  auto dtype() const -> py::dtype {
    return py::dtype("float32");
  }
};


template< typename F = float > 
struct PyDenseMatrix {
  py::array_t< F, py::array::c_style | py::array::forcecast > _data; 
  
  PyDenseMatrix(const py::array_t< F, py::array::c_style | py::array::forcecast >& data) : _data(data){
    // py::buffer_info buffer = data.request();
    // F* data = static_cast< F* >(buffer.ptr);
    // _data = std::vector(std::move(data), data + buffer.size);
  }
  
  // Needed to match the LinearOperator concept 
  auto matvec(const F* inp, F* out) const -> void {
    cMatrixOperations< F >::dense_matvec(
      _data.data(), inp,
      _data.shape(0), 
      _data.shape(1),
      true, // row-major 
      out
    );
  }
  auto matvec(const py::array_t< F >& input) const -> py::array_t< F > {
    auto out = std::vector< F >(static_cast< size_t >(_data.shape(0)), 0);
    this->matvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  auto shape() const -> pair< size_t, size_t > { 
    return std::make_pair(_data.shape(0), _data.shape(1));
  }

  auto dtype() const -> py::dtype {
    return py::dtype("float32");
  }
};


// TODO: use moves or buffer object to remove overhead
template< typename F > 
struct PyLinearOperator {
  using value_type = F;
  const py::object _op; 
  PyLinearOperator(const py::object& op) : _op(op) {
    if (!py::hasattr(op, "matvec")) { throw std::invalid_argument("Supplied object is missing 'matvec' attribute."); }
    if (!py::hasattr(op, "shape")) { throw std::invalid_argument("Supplied object is missing 'shape' attribute."); }
    // if (!op.has_attr("dtype")) { throw std::invalid_argument("Supplied object is missing 'dtype' attribute."); }
  }

  // Calls the matvec in python, casts the result to py::array_t, and copies through
  auto matvec(const F* inp, F* out) const {
    py::array_t< F > input({ static_cast<py::ssize_t>(shape().second) }, inp);
    py::object matvec_out = _op.attr("matvec")(input);
    py::array_t< F > output = matvec_out.cast< py::array_t< F > >();
    std::copy(output.data(), output.data() + output.size(), out);
  }

  auto matvec(const py::array_t< F >& input) const -> py::array_t< F > {
    auto out = std::vector< F >(static_cast< size_t >(shape().first), 0);
    this->matvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  auto shape() const -> pair< size_t, size_t > { 
    return _op.attr("shape").cast< std::pair< size_t, size_t > >();
  }

  auto dtype() const -> py::dtype {
    return py::dtype("float32");
  }
};

template< typename F > 
struct PyAdjointOperator {
  using value_type = F;
  const py::object _op; 
  PyAdjointOperator(const py::object& op) : _op(op) {
    if (!py::hasattr(op, "matvec")) { throw std::invalid_argument("Supplied object is missing 'matvec' attribute."); }
    if (!py::hasattr(op, "rmatvec")) { throw std::invalid_argument("Supplied object is missing 'matvec' attribute."); }
    if (!py::hasattr(op, "shape")) { throw std::invalid_argument("Supplied object is missing 'shape' attribute."); }
    // if (!op.has_attr("dtype")) { throw std::invalid_argument("Supplied object is missing 'dtype' attribute."); }
  }

  // Calls the matvec in python, casts the result to py::array_t, and copies through
  auto matvec(const F* inp, F* out) const {
    py::array_t< F > input({ static_cast<py::ssize_t>(shape().second) }, inp);
    py::object matvec_out = _op.attr("matvec")(input);
    py::array_t< F > output = matvec_out.cast< py::array_t< F > >();
    std::copy(output.data(), output.data() + output.size(), out);
  }
  auto matvec(const py::array_t< F >& input) const -> py::array_t< F > {
    auto out = std::vector< F >(static_cast< size_t >(shape().first), 0);
    this->matvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  // Calls the rmatvec in python, casts the result to py::array_t, and copies through
  void rmatvec(const F* inp, F* out) const {
    py::array_t< F > input({ static_cast<py::ssize_t>(shape().second) }, inp);
    py::object matvec_out = _op.attr("rmatvec")(input);
    py::array_t< F > output = matvec_out.cast< py::array_t< F > >();
    std::copy(output.data(), output.data() + output.size(), out);
  }
  auto rmatvec(const py::array_t< F >& input) const -> py::array_t< F > {
    auto out = std::vector< F >(static_cast< size_t >(shape().first), 0);
    this->rmatvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  auto shape() const -> pair< size_t, size_t > { 
    return _op.attr("shape").cast< std::pair< size_t, size_t > >();
  }

  auto dtype() const -> py::dtype {
    return py::dtype("float32");
  }
};

#endif 