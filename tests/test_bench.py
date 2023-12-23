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

def test_hutch_mf_bench_auto_gw(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, fun="log", maxiter=1500, quad="golub_welsch")
  assert True

def test_hutch_mf_bench_auto_fttr(benchmark):
  n = 100 
  A = symmetric(n)
  benchmark(hutch, A=A, fun="log", maxiter=1500, quad="fttr")
  assert True

def test_bench_quad_gw(benchmark):
  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  n = 100 
  A = symmetric(n)
  a, b = lanczos(A, deg=n)
  a, b = a, np.append([0], b)
  benchmark(_lanczos.quadrature, a, b, n, 0)
  assert True

def test_bench_quad_fttr(benchmark):
  from primate.diagonalize import lanczos, _lanczos
  np.random.seed(1234)
  n = 100 
  A = symmetric(n)
  a, b = lanczos(A, deg=n)
  a, b = a, np.append([0], b)
  benchmark(_lanczos.quadrature, a, b, n, 1)
  assert True


# def test_bench_toeplitz():
#   import timeit
#   from primate.operator import Toeplitz
#   from scipy.linalg import toeplitz

#   a = np.arange(1500)
#   Td = toeplitz(a)
#   Tm = Toeplitz(a)

#   x = np.random.uniform(size=len(a))
#   timeit.timeit(lambda: Td @ x, number = 1000)
#   timeit.timeit(lambda: Tm @ x, number = 1000)
  
#   sys.getsizeof(Td)
#   getsize(Tm)
#   72824 / 18000128 


