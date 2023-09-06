#ifndef _RANDOM_GENERATOR_RANDOM_CONCEPTS_H
#define _RANDOM_GENERATOR_RANDOM_CONCEPTS_H

#include <concepts>
#include <random> // SeedSequence

// std::initializer_list< std::uint32_t >
template < typename T >
concept LightRandom64Engine = requires(T rne, std::seed_seq& S) {
  T::state_size > 0;                            // has a known state size, in tersm of number 32 unsigned integers
  { rne() } -> std::same_as< std::uint64_t >;                   // <engine>() yields an unsigned 64-bit integer 
  { rne.seed(S) };                       // state is seedeable w/ pointer of integers
} && std::default_initializable< T >;

// See: https://stackoverflow.com/questions/39288595/why-not-just-use-random-device/
template < typename T >
concept ThreadSafeRBG = requires(T rbg, int tid) {
  { rbg.initialize(tid) };
  { rbg.next(tid) } -> std::convertible_to< std::uint_fast64_t >;
};
   

#endif