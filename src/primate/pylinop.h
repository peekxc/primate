#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

template< std::floating_point F > 
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
    py_array< F > input({ static_cast<py::ssize_t>(shape().second) }, inp);
    py::object matvec_out = _op.attr("matvec")(input);
    py::array_t< F > output = matvec_out.cast< py::array_t< F > >();
    std::copy(output.data(), output.data() + output.size(), out);
  }

  auto matvec(const py_array< F >& input) const -> py_array< F > {
    auto out = std::vector< F >(static_cast< size_t >(shape().first), 0);
    this->matvec(input.data(), static_cast< F* >(&out[0]));
    return py::cast(out);
  }

  auto shape() const -> pair< size_t, size_t > { 
    return _op.attr("shape").cast< std::pair< size_t, size_t > >();
  }

  auto dtype() const -> py::dtype {
    auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
    return dtype;
  }
};