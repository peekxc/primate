#ifndef _PYLINOP_H 
#define _PYLINOP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

#include <chrono>
using us = std::chrono::microseconds;
using dur_seconds = std::chrono::duration< double >;
using hr_clock = std::chrono::high_resolution_clock;

template< std::floating_point F > 
struct PyLinearOperator {
  using value_type = F;
  const py::object _op; 
  mutable size_t matvec_time; 
	std::pair< size_t, size_t > _shape; // copy the shape on construct
  
  PyLinearOperator(const py::object op) : _op(op), matvec_time(0.0) {
    if (!py::hasattr(op, "matvec")) { throw std::invalid_argument("Supplied object is missing 'matvec' attribute."); }
    if (!py::hasattr(op, "shape")) { throw std::invalid_argument("Supplied object is missing 'shape' attribute."); }
    // if (!op.has_attr("dtype")) { throw std::invalid_argument("Supplied object is missing 'dtype' attribute."); }
		_shape = _op.attr("shape").template cast< std::pair< size_t, size_t > >();
    matvec_time = 0; 
  }

  // Calls the matvec in python, casts the result to py::array_t, and copies through
  auto matvec(const F* inp, F* out) const {
    auto ts = hr_clock::now(); 
    py_array< F > input({ static_cast<py::ssize_t>(_shape.second) }, inp);
    py::object matvec_out = _op.attr("matvec")(input);
    py::array_t< F > output = matvec_out.cast< py::array_t< F > >();
    // std::copy(output.data(), output.data() + output.size(), out);
		std::copy(output.data(), output.data() + _shape.first, out);
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  // auto matmat(const F* X, F* Y) const {
  //   if (py::hasattr(op, "matvec")){ 
  //     return; 
  //   } else {

  //   }
  //   Eigen::Map< const DenseMatrix< F > > XM(X, op->shape().second, k);
  //   Eigen::Map< DenseMatrix< F > > YM(Y, op->shape().first, k);
  //   for (size_t j = 0; j < k; ++j){
  //     matvec(XM.col(j).data(), YM.col(j).data());
  //   }
  //   // py_array< F > input({ static_cast<py::ssize_t>(shape().second) }, inp);
  //   // py::object matvec_out = _op.attr("matvec")(input);
  //   // py::array_t< F > output = matvec_out.cast< py::array_t< F > >();
  //   // std::copy(output.data(), output.data() + output.size(), out);
  // }

  auto matvec(const py_array< F >& input) const -> py_array< F > {
    auto out = std::vector< F >(static_cast< size_t >(shape().first), 0);
    this->matvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  auto shape() const -> pair< size_t, size_t > { 
    return _op.attr("shape").template cast< std::pair< size_t, size_t > >();
  }

  auto dtype() const -> py::dtype {
    auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
    return dtype;
  }
};

#endif