#ifndef SPECTRAL_FUNCTIONS_H
#define SPECTRAL_FUNCTIONS_H

#include <concepts> 
#include <string> 
#include <functional>  // function, identity
#include <unordered_map> // unordered_map 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

// py::function g = kwargs_map["matrix_func"].cast< py::function >(); 
//   // f = [g](F val) -> F { return g(val).template cast< F >(); }; // make sure to copy g
//   f = [N, g](F* eigenvalues){ 
//     std::vector< F > ew = std::vector< F >(eigenvalues, eigenvalues+N);
//     std::vector< F > ew_new = g(ew).template cast< std::vector< F > >();
//     if (ew_new.size() != ew.size()){
//       throw std::invalid_argument("Dimension mismatch. Matrix function must return vector of same length as input.");
//     }
//     std::copy(ew_new.begin(), ew_new.end(), eigenvalues);
//   };
// using fn_type = void(*)(F*);
// const auto *result = f.template target< fn_type >();
// if (!result) {
//   std::cout << "Failed to convert to function ptr! Falling back to pyfunc" << std::endl;
//   // M.f = [fun](F x) -> F { return fun(x).template cast< F >(); };
//   M.f = [fun](F* x) -> void { fun(x); };
// } else {
//   std::cout << "Native cpp function detected!" << std::endl;
//   M.f = f; 
// }
// std::function< void(F*) > f = py::cast< std::function< void(F*) > >(fun);
// using fn_type = void(*)(F*);
// const auto *result = f.template target< fn_type >();
// if (!result) {
//   std::cout << "Failed to convert to function ptr! Falling back to pyfunc" << std::endl;
//   // M.f = [fun](F x) -> F { return fun(x).template cast< F >(); };
//   M.f = [fun](F* x) -> void { fun(x); };
// } else {
//   std::cout << "Native cpp function detected!" << std::endl;
//   M.f = f; 
// }

template< std::floating_point F >
auto deduce_vector_func_from_scalar(const py::function fun, const size_t N, const py::kwargs& kwargs) -> std::optional< std::function< void(F*) > > {
  // See also: https://github.com/pybind/pybind11/blob/master/tests/test_callbacks.cpp
  std::function< F(F) > f;
  try {
    f = py::cast< std::function< F(F) > >(fun);
  } catch(py::cast_error& e){
    std::cout << "Failed to deduce function as scalar-valued!" << std::endl;
    return std::nullopt;
  }
  // Copy the deduced function to a new one and return it
  std::function< void(F*) > g = [f, N](F* eigenvalues){
    std::for_each_n(eigenvalues, N, f); // [](auto ew) -> F { return std::log(ew); })
    return;
  };
  return std::make_optional< std::function< void(F*) > >(g);
}
    
template< std::floating_point F >
auto deduce_vector_func(const py::function fun, bool& native) -> std::optional< std::function< void(F*, const size_t) > > {
  // See also: https://github.com/pybind/pybind11/blob/master/tests/test_callbacks.cpp
  // std::function< void(F*, const size_t) > f;
  
  // First try to deduce a raw C function pointer
  auto f = py::cast< std::function< void(F*, const size_t) > >(fun);
  using fn_type = void(*)(F*, const size_t);
  const auto *result = f.template target< fn_type >();
  if (result) {
    native = true; 
    // std::cout << "Native cpp function detected!" << std::endl;
    return std::make_optional(f); // copy the ptr; this should avoid C++ -> Python -> C++ roundtrips
  } else {
    // std::cout << "Failed to convert to function ptr! Falling back to pyfunc" << std::endl;
    try {
      using arr_t = py::array_t< F >;
      const auto g = py::cast< std::function< arr_t(arr_t&) > >(fun);
      f = [g](F* eigenvalues, const size_t N){
        auto inp = py::array_t< F >(N, eigenvalues);
        auto out = g(inp);
        std::copy(out.data(), out.data() + std::min(size_t(out.size()), N), eigenvalues);
        return;
      };
      native = false;  
    } catch(py::cast_error& e){
      std::cout << "Failed to deduce apprioriate function! Is it vector-valued?" << std::endl;
      return std::nullopt;
    }
    // std::cout << "Function detected as vector-valued function!" << std::endl;
    return std::make_optional< std::function< void(F*, const size_t) > >(f); 
  }
}
//std::vector< F > ew_new = fun(ew).template cast< std::vector< F > >();

template< std::floating_point F > 
auto param_spectral_func(const py::kwargs& kwargs) -> std::function< F(F) >{
  constexpr F SQRT2_REC = 0.70710678118654752440;  // 1 / sqrt(2)
  constexpr F SQRT2_RECPI = 1.12837916709551257390; // 2 / sqrt(pi)
  auto kwargs_map = kwargs.cast< std::unordered_map< std::string, py::object > >();
  std::function< F(F) > f = std::identity();
  if (kwargs_map.contains("function")){
    std::string matrix_func = kwargs_map["function"].cast< std::string >(); // py::function
    if (matrix_func == "identity"){
      f = std::identity(); 
    } else if (matrix_func == "abs"){
      f = [](F eigenvalue) -> F { return std::abs(eigenvalue); }; 
    } else if (matrix_func == "sqrt"){
      f = [](F eigenvalue) -> F { return std::sqrt(std::abs(eigenvalue)); }; 
    } else if (matrix_func == "log"){
      f = [](F eigenvalue) -> F { return std::log(eigenvalue); }; 
    } else if (matrix_func == "inv"){
      f = [](F eigenvalue) -> F {  return 1.0/eigenvalue; };
    } else if (matrix_func == "exp"){
      F t = kwargs_map.contains("t") ? kwargs_map["t"].cast< F >() : 1.0;
      f = [t](F eigenvalue) -> F {  return std::exp(t*eigenvalue); };  
    } else if (matrix_func == "smoothstep"){
      F a = kwargs_map.contains("a") ? kwargs_map["a"].cast< F >() : 0.0;
      F b = kwargs_map.contains("b") ? kwargs_map["b"].cast< F >() : 1.0;
      // const bool pos_only = kwargs_map.contains("pos") ? kwargs_map["pos"].cast< bool >() : false;
      const F d = (b-a);
      f = [a, d](F eigenvalue) -> F { 
        F y = std::clamp(std::abs(F((eigenvalue-a)/d)), F(0.0), F(1.0));
        y = 3.0 * std::pow(y, 2) - 2.0 * std::pow(y, 3);
        return y;
      };
    } else if (matrix_func == "gaussian"){
      F mu = kwargs_map.contains("mu") ? kwargs_map["mu"].cast< F >() : 0.0;
      F sigma = kwargs_map.contains("sigma") ? kwargs_map["sigma"].cast< F >() : 1.0;
      f = [mu, sigma](F eigenvalue) -> F {  
        auto x = (mu - eigenvalue) / sigma;
        return (0.5 * SQRT2_REC * SQRT2_RECPI / sigma) * exp(-0.5 * x * x); 
      }; 
    } else if (matrix_func == "numrank"){
      F threshold = kwargs_map.contains("threshold") ? kwargs_map["threshold"].cast< F >() : 0.000001;
      f = [threshold](F eigenvalue) -> F {  
        return std::abs(eigenvalue) > threshold ? F(1.0) : F(0.0);
      };  
    } else if (matrix_func == "generic"){
      if (kwargs_map.contains("matrix_func")){
        py::function g = kwargs_map["matrix_func"].cast< py::function >(); 
        f = [g](F val) -> F { return g(val).template cast< F >(); }; // make sure to copy g
      } else {
        f = std::identity();
      }
    } else {
      throw std::invalid_argument("Invalid matrix function supplied");
    }
  } else {
    throw std::invalid_argument("No matrix function supplied.");
  }
  return f; 
}

// Vector-version
template< std::floating_point F > 
auto param_vector_func(const py::kwargs& kwargs, bool& native) -> std::function< void(F*, const size_t) >{
  constexpr F SQRT2_REC = 0.70710678118654752440;  // 1 / sqrt(2)
  constexpr F SQRT2_RECPI = 1.12837916709551257390; // 2 / sqrt(pi)
  auto kwargs_map = kwargs.cast< std::unordered_map< std::string, py::object > >();
  if (!kwargs_map.contains("function")){ 
    throw std::invalid_argument("No matrix function supplied.");
  }

  // Deduce the matrix function 
  std::function< void(F*, const size_t) > f; 
  std::string matrix_func = kwargs_map["function"].cast< std::string >(); // py::function
  if (matrix_func == "identity"){
    f = [](F* eigenvalues, [[maybe_unused]] const size_t N){ return; };
  } else if (matrix_func == "abs"){
    f = [](F* eigenvalues, const size_t N){ 
      std::for_each_n(eigenvalues, N, [](auto& ew) { ew = std::abs(ew); }); 
    };
  } else if (matrix_func == "sqrt"){
    f = [](F* eigenvalues, const size_t N){ 
      std::for_each_n(eigenvalues, N, [](auto& ew) { ew = std::abs(ew); }); 
      std::for_each_n(eigenvalues, N, [](auto& ew) { ew = std::sqrt(ew); }); 
    };
    // f = [](F eigenvalue) -> F { return std::sqrt(std::abs(eigenvalue)); }; 
  } else if (matrix_func == "log"){
    f = [](F* eigenvalues, const size_t N) -> void { 
      std::for_each_n(eigenvalues, N, [](auto& ew) { ew = std::log(ew); });
    }; 
  } else if (matrix_func == "inv"){
    f = [](F* eigenvalues, const size_t N) -> void { 
      std::for_each_n(eigenvalues, N, [](auto& ew) { ew = 1 / ew; });
    };
  } else if (matrix_func == "exp"){
    F t = kwargs_map.contains("t") ? kwargs_map["t"].cast< F >() : 1.0;
    f = [t](F* eigenvalues, const size_t N){ 
      std::for_each_n(eigenvalues, N, [t](F& ew) { ew = std::exp(t * ew); });
    };
  } else if (matrix_func == "smoothstep"){
    F a = kwargs_map.contains("a") ? kwargs_map["a"].cast< F >() : 0.0;
    F b = kwargs_map.contains("b") ? kwargs_map["b"].cast< F >() : 1.0;
    // const bool pos_only = kwargs_map.contains("pos") ? kwargs_map["pos"].cast< bool >() : false;
    const F d = (b-a);
    // f = [a, d](F eigenvalue) -> F { 
    //   F y = std::clamp(std::abs(F((eigenvalue-a)/d)), F(0.0), F(1.0));
    //   y = 3.0 * std::pow(y, 2) - 2.0 * std::pow(y, 3);
    //   return y;
    // };
    f = [a, d](F* eigenvalues, const size_t N){ 
      for (size_t i = 0; i < N; ++i){ 
        F y = std::clamp(std::abs(F((eigenvalues[i]-a)/d)), F(0.0), F(1.0));
        eigenvalues[i] = 3.0 * std::pow(y, 2) - 2.0 * std::pow(y, 3);
      }
    };
  } else if (matrix_func == "gaussian"){
    F mu = kwargs_map.contains("mu") ? kwargs_map["mu"].cast< F >() : 0.0;
    F sigma = kwargs_map.contains("sigma") ? kwargs_map["sigma"].cast< F >() : 1.0;
    f = [sigma, mu](F* eigenvalues, const size_t N){ 
      for (size_t i = 0; i < N; ++i){ 
        auto x = (mu - eigenvalues[i]) / sigma;
        eigenvalues[i] = (0.5 * SQRT2_REC * SQRT2_RECPI / sigma) * exp(-0.5 * x * x);
      }
    };
  } else if (matrix_func == "numrank"){
    F threshold = kwargs_map.contains("threshold") ? kwargs_map["threshold"].cast< F >() : 0.000001;
    f = [threshold](F* eigenvalues, const size_t N){ 
      std::for_each_n(eigenvalues, N, [threshold](auto& ew) { ew = std::abs(ew) > threshold ? F(1.0) : F(0.0); });
    };
  } else if (matrix_func == "generic" && kwargs_map.contains("matrix_func")){
    const py::function g = kwargs_map["matrix_func"].cast< const py::function >(); 
    
    // Try to deduce user-supplied vector-valued function
    auto ov = deduce_vector_func< F >(g, native);
    if (ov){ f = *std::move(ov); return f; }
    
    // Try to deduce scalar valued 
    // auto os = deduce_vector_func_from_scalar< F >(g, N, kwargs);
    // if (os){ f = *std::move(os); return f; }

    throw std::invalid_argument("Invalid matrix function supplied. Must be defined for vector inputs.");
  } else {
    throw std::invalid_argument("Invalid matrix function supplied");
  }
  native = true; 
  return f; 
}

// TODO: add param transform function with deflation options


#endif