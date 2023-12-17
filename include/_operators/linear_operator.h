
#ifndef _OPERATORS_H
#define _OPERATORS_H

#include <type_traits>
#include <concepts>
#include <algorithm>
#include <functional>

using std::pair; 
using std::vector; 

template <typename T>
struct TypeString;

template <> struct TypeString< float > {
  static constexpr const char* value = "float32";
};

template <> struct TypeString< double > {
  static constexpr const char* value = "float64";
};

// Generalizes concepts from: https://ameli.github.io/imate/generated/imate.Matrix.html#imate-matrix
template < typename T, typename F = typename T::value_type >
concept LinearOperator = requires(T op, const F* input, F* output) {
  { op.matvec(input, output) }; // y = A x 
  { op.shape() } -> std::convertible_to< std::pair< size_t, size_t > >;
};

template < typename T, typename F = typename T::value_type >
concept AdjointOperator = requires(T op, const F* input, F* output) {
  { op.rmatvec(input, output) }; // y = A^T x 
} && LinearOperator< T, F >;

template < typename T, typename F = typename T::value_type >
concept LinearAdditiveOperator = requires(T op, const F* input, const F alpha, F* output) {
  { op.matvec_add(input, alpha, output) }; // y = y + \alpha * Ax
} && LinearOperator< T, F >;

template < typename T, typename F = typename T::value_type >
concept AdjointAdditiveOperator = requires(T op, const F* input, const F alpha, F* output) {
  { op.rmatvec_add(input, alpha, output) }; // y = y + \alpha * A^T x
} && AdjointOperator< T, F >;

template < typename T, typename F = typename T::value_type >
concept Operator = 
  LinearOperator< T, F > || 
  AdjointOperator< T, F > || 
  LinearAdditiveOperator< T, F > ||
  AdjointAdditiveOperator< T, F >
;

template < typename T, typename F = typename T::value_type >
concept QuadOperator = requires(T op, const F* input) {
  { op.quad(input) } -> std::convertible_to< F >; // v^T A v ; todo: make optional if matvec available?
};

template < typename T, typename F = typename T::value_type >
concept SupportsMatrixMult = requires(T op, const F* input, F* output, const int k) {
  { op.matmat(input, output, k) };
};

// Represents the operator (A + tB) for parameter T = { t1, t2, ..., tn }
template < typename T, typename F = typename T::value_type >
concept AffineOperator = requires(T op, F t) {
  T::relation_known; // -> std::convertible_to< bool >;
  { op.set_parameter(t) };
  // { op.get_num_parameters() } -> std::convertible_to< size_t >;
} && Operator< T, F >;

#endif