# primate 

`primate`, short for **Pr**obabalistic **I**mplicit **Ma**trix **T**race **E**stimator, is Python package that performs randomized matrix trace estimation of [matrix functions](https://en.wikipedia.org/wiki/Analytic_function_of_a_matrix); that is, matrices parameterized by functions:

$$ \mathrm{tr}(f(A)) \triangleq \mathrm{tr}(U f(\Lambda) U^{\intercal}), \quad \quad f : [a,b] \to \mathbb{R}$$

Trace estimates are obtained in a Monte-Carlo fashion via the _stochastic Lanczos method_ (SLQ)[^1]. This method is useful for sparse or highly structured matrices with efficiently computable [quadratic forms](https://en.wikipedia.org/wiki/Quadratic_form#Associated_symmetric_matrix).

Notable features of `primate` include:

- A highly-parametrizable trace estimator (SLQ)
- Various distribution / engine choices for random vector generation (the stochastic part!)
- Orthogonalization routines, such as the _Lanczos_, _Golub Kahan_, and _Gram Schmidt_ methods
- Support for _arbitrary_ matrix functions, i.e. `Callable`'s (Python) and `invocable`'s[^2] (C++)
- Support for _arbitrary_ `LinearOperator`'s, e.g. those in [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy-sparse-linalg-linearoperator) or [Pylops](https://pylops.readthedocs.io/en/stable/index.html)

<!-- Moreover, `primate`'s C++ API uses a generic template interface written with [C++20 Concepts](https://en.cppreference.com/w/cpp/language/constraints)---thus, any `LinearOperator` [fitting the constraints](https://github.com/peekxc/primate/blob/d09459c017fcba68a11eaeb56296ef0c97d6c053/include/_linear_operator/linear_operator.h#L21-L49).  -->
<!-- To use,, the library is is [header-only](https://en.wikipedia.org/wiki/Header-only), so integration is a si.  -->

Much of `primate`'s computational code was directly ported from the (excellent) [`imate` package](https://github.com/ameli/imate)---for a comparison of the two, see [here](imate_compare.qmd).

[^1]: Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
[^2]: This includes [std::function](https://en.cppreference.com/w/cpp/utility/functional/function)'s, C-style function pointers, [functors](https://stackoverflow.com/questions/356950/what-are-c-functors-and-their-uses), and [lambda expressions](https://en.cppreference.com/w/cpp/language/lambda).




