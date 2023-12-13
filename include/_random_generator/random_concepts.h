#ifndef _RANDOM_GENERATOR_RANDOM_CONCEPTS_H
#define _RANDOM_GENERATOR_RANDOM_CONCEPTS_H

#include <concepts>
#include <random> // SeedSequence

// std::initializer_list< std::uint32_t >
template < typename T >
concept LightRandom64Engine = requires(T rne, std::seed_seq& S) {
  { rne() } -> std::same_as< std::uint64_t >;   // <engine>() yields an unsigned 64-bit integer 
  { rne.seed(S) };                              // state is seedeable w/ pointer of integers
} && std::default_initializable< T > && std::uniform_random_bit_generator< T >;

template < typename T >
concept Stateful64Engine = requires(T rne) {
  T::state_size > 0;                            // has a known positive state size, in terms of number 32 unsigned integers
} && LightRandom64Engine< T >;

// See: https://stackoverflow.com/questions/39288595/why-not-just-use-random-device/
template < typename T >
concept ThreadSafeRBG = requires(T rbg, int tid) {
  { rbg.initialize(tid) };
  { rbg.next(tid) } -> std::convertible_to< std::uint_fast64_t >;
};
   
struct Random64EngineConcept {
  using result_type = uint64_t;
  // virtual const void seed(int) const = 0;
  virtual ~Random64EngineConcept() = default;
  virtual void seed(std::seed_seq&) = 0;
  virtual uint64_t operator()() = 0; 
  virtual size_t state_size() const = 0;  
  static constexpr uint64_t min() { return std::numeric_limits< uint64_t >::min(); }
  static constexpr uint64_t max() { return std::numeric_limits< uint64_t >::max(); }
};

template < LightRandom64Engine T >
struct Random64Engine : public Random64EngineConcept {
  using result_type = uint64_t;
  T rng;
  Random64Engine() : rng(T()) { }
  void seed(std::seed_seq& S) override { rng.seed(S); };
  uint64_t operator()() override { return rng(); }
  size_t state_size() const override {
    if constexpr (Stateful64Engine< T >){ return std::max(T::state_size, size_t(1)); } 
    return 1; 
  }
  // T* get_rng(){ return &rng; } 
};

#endif