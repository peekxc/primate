`primate`, short for **Pr**obabalistic **I**mplicit **Ma**trix **T**race **E**stimator, is a Python package that provides estimators of quantities derived from [matrix functions](https://en.wikipedia.org/wiki/Analytic_function_of_a_matrix); that is, matrices parameterized by functions:

$$ f(A) \triangleq U f(\Lambda) U^{\intercal}, \quad \quad f : [a,b] \to \mathbb{R}$$

Matrix function approximations are obtained via the _Lanczos_[^1] and _stochastic Lanczos quadrature_[^2] methods, which are well-suited for sparse or highly structured operators with fast [quadratic forms](https://en.wikipedia.org/wiki/Quadratic_form#Associated_symmetric_matrix).

Notable features of `primate` include:

- Efficient methods for trace, quadrature, and matrix function approximation
- Various distribution / engine choices for random vector generation (the stochastic part!)
- Support for _arbitrary_ matrix functions, i.e. `Callable`'s (Python) and `invocable`'s[^3] (C++)
- Support for _arbitrary_ `LinearOperator`'s, e.g. those in [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy-sparse-linalg-linearoperator) or [Pylops](https://pylops.readthedocs.io/en/stable/index.html)
- Matrix-free interface to the _Lanczos_, _Golub-Welsch_, and _Gram Schmidt_ methods

<!-- Moreover, `primate`'s C++ API uses a generic template interface written with [C++20 Concepts](https://en.cppreference.com/w/cpp/language/constraints)---thus, any `LinearOperator` [fitting the constraints](https://github.com/peekxc/primate/blob/d09459c017fcba68a11eaeb56296ef0c97d6c053/include/_linear_operator/linear_operator.h#L21-L49).  -->
<!-- To use,, the library is is [header-only](https://en.wikipedia.org/wiki/Header-only), so integration is a si.  -->

`primate` was partially inspired by the [`imate` package](https://github.com/ameli/imate)---for a comparison of the two, see [here](https://peekxc.github.io/primate/imate_compare.html).

[^1]: Musco, Cameron, Christopher Musco, and Aaron Sidford. "Stability of the Lanczos method for matrix function approximation." Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics, 2018.
[^2]: Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature.
[^3]: This includes [std::function](https://en.cppreference.com/w/cpp/utility/functional/function)'s, C-style function pointers, [functors](https://stackoverflow.com/questions/356950/what-are-c-functors-and-their-uses), and [lambda expressions](https://en.cppreference.com/w/cpp/language/lambda).

