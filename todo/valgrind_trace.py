# import faulthandler
# faulthandler.enable()
import numpy as np
from imate import toeplitz, schatten, trace
from primate import slq
A = toeplitz(2, 1, size=100, gram=True).astype(np.float32).tocsc()

# %% 
tr_est = slq(A, gram=True, num_threads=1)
print(tr_est)

tr_est, info = slq(A, gram=False, num_threads=1, return_info=True)

print(np.squeeze(info['convergence']['samples']))

from scipy.sparse import random
A = random(10, 5, density=0.15).astype(np.float32).tocsc()
tr_est = slq(A, gram=False, num_threads=1)
assert not(np.isnan(np.take(tr_est, 0)))

sum(A @ A.T).diagonal()
# trace(A, gram=False, p=2.5, method='slq')
# PYTHONFAULTHANDLER=1 PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=full python -q -X faulthandler examples/valgrind_trace.py 
