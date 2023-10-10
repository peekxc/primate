import numpy as np 
from pyimate.trace import _trace
import timeit 

x = np.random.uniform(size=1500000).astype(np.float64)

timeit.timeit(lambda: _trace.parallel_computation(x, 1), number=1000)
timeit.timeit(lambda: _trace.parallel_computation(x, 2), number=1000)
timeit.timeit(lambda: _trace.parallel_computation(x, 4), number=1000)
timeit.timeit(lambda: _trace.parallel_computation(x, 6), number=1000)

# import timeit
# import pyimate
# import imate
# from pyimate.trace import trace_estimator
# A = imate.toeplitz(2, 1, size=1500, gram=True) # gram = True for symmetric
# tr_est = trace_estimator(A, info=False, num_threads=2, min_num_samples=50, max_num_samples=100)

# timeit.timeit(lambda: trace_estimator(A, info=False, num_threads=1, min_num_samples=150, max_num_samples=200), number=100)
# timeit.timeit(lambda: trace_estimator(A, info=False, num_threads=2, min_num_samples=150, max_num_samples=200), number=100)
# timeit.timeit(lambda: trace_estimator(A, info=False, num_threads=4, min_num_samples=150, max_num_samples=200), number=100)
# timeit.timeit(lambda: trace_estimator(A, info=False, num_threads=6, min_num_samples=150, max_num_samples=200), number=100)