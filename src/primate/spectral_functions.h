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
auto param_vector_func(const size_t N, const py::kwargs& kwargs) -> std::function< void(F*) >{
  constexpr F SQRT2_REC = 0.70710678118654752440;  // 1 / sqrt(2)
  constexpr F SQRT2_RECPI = 1.12837916709551257390; // 2 / sqrt(pi)
  auto kwargs_map = kwargs.cast< std::unordered_map< std::string, py::object > >();
  std::function< void(F*) > f; 
  if (kwargs_map.contains("function")){
    std::string matrix_func = kwargs_map["function"].cast< std::string >(); // py::function
    if (matrix_func == "identity"){
      f = [](F* eigenvalues){ return; };
    } else if (matrix_func == "abs"){
      f = [N](F* eigenvalues){ 
        std::for_each_n(eigenvalues, N, [](auto ew) -> F { return std::abs(ew); }); 
      };
      // f = [](F eigenvalue) -> F { return std::abs(eigenvalue); }; 
    } else if (matrix_func == "sqrt"){
      // const auto g = [](F ew) -> F { return std::sqrt(std::abs(ew)); };
      f = [N](F* eigenvalues){ 
        std::for_each_n(eigenvalues, N, [](auto ew) -> F { return std::abs(ew); }); 
        std::for_each_n(eigenvalues, N, [](auto ew) -> F { return std::sqrt(ew); }); 
        // std::transform(eigenvalues, eigenvalues + N, eigenvalues, std::abs);
        // std::transform(eigenvalues, eigenvalues + N, eigenvalues, std::sqrt);
      };
      // f = [](F eigenvalue) -> F { return std::sqrt(std::abs(eigenvalue)); }; 
    } else if (matrix_func == "log"){
      f = [N](F* eigenvalues) -> F { 
        std::for_each_n(eigenvalues, N, [](auto ew) -> F { return std::log(ew); });
      }; 
    } else if (matrix_func == "inv"){
      // f = [](F eigenvalue) -> F {  return 1.0/eigenvalue; };
      f = [N](F* eigenvalues){ 
        std::for_each_n(eigenvalues, N, [](auto ew) -> F { return 1 / ew; });
      };
    } else if (matrix_func == "exp"){
      F t = kwargs_map.contains("t") ? kwargs_map["t"].cast< F >() : 1.0;
      // f = [t](F eigenvalue) -> F {  return std::exp(t*eigenvalue); };  
      f = [N, t](F* eigenvalues){ 
        std::for_each_n(eigenvalues, N, [t](auto ew) -> F { return std::exp(t * ew); });
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
      f = [N, a, d](F* eigenvalues){ 
        for (size_t i = 0; i < N; ++i){ 
          F y = std::clamp(std::abs(F((eigenvalues[i]-a)/d)), F(0.0), F(1.0));
          eigenvalues[i] = 3.0 * std::pow(y, 2) - 2.0 * std::pow(y, 3);
        }
      };
    } else if (matrix_func == "gaussian"){
      F mu = kwargs_map.contains("mu") ? kwargs_map["mu"].cast< F >() : 0.0;
      F sigma = kwargs_map.contains("sigma") ? kwargs_map["sigma"].cast< F >() : 1.0;
      // f = [mu, sigma](F eigenvalue) -> F {  
      //   auto x = (mu - eigenvalue) / sigma;
      //   return (0.5 * SQRT2_REC * SQRT2_RECPI / sigma) * exp(-0.5 * x * x); 
      // }; 
      f = [N, sigma, mu](F* eigenvalues){ 
        for (size_t i = 0; i < N; ++i){ 
          auto x = (mu - eigenvalues[i]) / sigma;
          eigenvalues[i] = (0.5 * SQRT2_REC * SQRT2_RECPI / sigma) * exp(-0.5 * x * x);
        }
      };
    } else if (matrix_func == "numrank"){
      F threshold = kwargs_map.contains("threshold") ? kwargs_map["threshold"].cast< F >() : 0.000001;
      // f = [threshold](F eigenvalue) -> F {  
      //   return std::abs(eigenvalue) > threshold ? F(1.0) : F(0.0);
      // };  
      f = [N, threshold](F* eigenvalues){ 
        std::for_each_n(eigenvalues, N, [threshold](auto ew) -> F { std::abs(ew) > threshold ? F(1.0) : F(0.0); });
      };
    } else if (matrix_func == "generic"){
      if (kwargs_map.contains("matrix_func")){
        py::function g = kwargs_map["matrix_func"].cast< py::function >(); 
        // f = [g](F val) -> F { return g(val).template cast< F >(); }; // make sure to copy g
        f = [N, g](F* eigenvalues){ 
          std::vector< F > ew = std::vector< F >(eigenvalues, eigenvalues+N);
          std::vector< F > ew_new = g(ew).template cast< std::vector< F > >();
          if (ew_new.size() != ew.size()){
            throw std::invalid_argument("Dimension mismatch. Matrix function must return vector of same length as input.");
          }
          std::copy(ew_new.begin(), ew_new.end(), eigenvalues);
        };
      } else {
        f = [](F* eigenvalues){ return; };
      }
    } else {
      throw std::invalid_argument("Invalid matrix function supplied");
    }
  } else {
    throw std::invalid_argument("No matrix function supplied.");
  }
  return f; 
}

#endif