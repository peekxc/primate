~ Under Construction ~ 


<!-- # primate 

`primate`, short for **Pr**obabalistic **I**mplicit **Ma**trix **T**race **E**stimator, is Python package for randomized matrix trace estimation. The package contains a variety of functions largely geared towards implicit trace estimation of [matrix functions](https://en.wikipedia.org/wiki/Analytic_function_of_a_matrix#Classes_of_matrix_functions) vis-a-vi the _stochastic Lanczos method_[^1]. In particular, `primate` offers:

- Trace estimation for _arbitrary_ matrix functions, supplied as `Callables` (via Python) or `std::invocable`'s (via C++) (see details)
- Support for _arbitrary_ `LinearOperator`'s, e.g. those in SciPy or Pylops 
- Diagonalization and orthogonalization routines, such as the _Lanczos_ and _Gram Schmidt_ methods
- Various distribution/engine choices for random vector generation (the stochastic part!)

Moreover, `primate`'s core C++ API is exported as a [header-only library](https://en.wikipedia.org/wiki/Header-only) and uses a generic template interface via [C++20 Concepts](https://en.cppreference.com/w/cpp/language/constraints)---thus, one can easily import and extend the various linear algebra routines in other Python/C++ projects by just adding the right `#include`'s and supplying types [fitting the constraints](https://github.com/peekxc/primate/blob/d09459c017fcba68a11eaeb56296ef0c97d6c053/include/_linear_operator/linear_operator.h#L21-L49). This makes it incredibly easy to e.g. add a non-standard matrix function or compile the trace estimator with custom linear operator (todo: document this).

Most of `primate`'s core computational code was directly ported from the (excellent) [`imate` package](https://github.com/ameli/imate). In comparison to `imate`, `primate` was developed with slightly different goals in mind, most of which have to do with integrability and the choice of FFI / build system. 

Some notable differences between the two packages:  

- `imate` supports multiple trace estimation approaches, e.g. decompositional methods. `primate` only supports the SLQ method.  
- `imate` supports both CPU parallelism and GPU parallelism. `primate` only supports CPU parallelism.
- `imate` uses setuptools to build Cython bindings, whereas `primate` uses [meson-python](https://meson-python.readthedocs.io/en/latest/) to build [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) bindings
- `imate` internal libraries are dynamically linked, wheres `primate`'s C++ is header-only ([what's the difference?](https://stackoverflow.com/questions/12671383/benefits-of-header-only-libraries))
- `imate` works with any standard C++ compiler, whereas `primate` requires (some) compiler support for C++20.

In general, if your needs are satisfied by `imate`, you should prefer it! However, if you need support for one of the things mentioned above, perhaps try `primate`!

[1]: Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
 -->


