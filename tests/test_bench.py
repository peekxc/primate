import numpy as np 
from primate.random import symmetric
from primate.trace import hutch

def test_hutch_bench_1(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, maxiter=1500, num_threads=1)
  assert True

def test_hutch_bench_2(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, maxiter=1500, num_threads=2)
  assert True

def test_hutch_bench_4(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, maxiter=1500, num_threads=4)
  assert True

def test_hutch_bench_8(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, maxiter=1500, num_threads=4)
  assert True

def test_hutch_bench_auto(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, maxiter=1500)
  assert True