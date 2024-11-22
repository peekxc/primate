from array import array
from numbers import Number
from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .fttr import fttr
from .lanczos import lanczos
from .tridiag import eigh_tridiag, eigvalsh_tridiag


def lanczos_quadrature(
	d: np.ndarray,
	e: np.ndarray,
	deg: Optional[int] = None,
	quad: str = "gw",  # The method of computing the weights
	nodes: Optional[np.ndarray] = None,  # Output nodes of the quadrature
	weights: Optional[np.ndarray] = None,  # Output weights of the quadrature
	**kwargs: dict,
):
	r"""Compute the Gaussian quadrature rule of a tridiagonal Jacobi matrix.

	This function computes the degree-$d$ (`deg`) Gaussian quadrature rule for a symmetric Jacobi matrix $J$,
	which associates `nodes` to the eigenvalues of $J$ and `weights` to the squares of the first components
	of the eigenvectors of $J$. The resulting rule is a weighted sum approximating the definite integral:

	$$ \int_{a}^{b} f(x) \omega(x) dx \approx \sum\limits_{i=1}^d f(x_i) w_i $$

	where $\omega(x)$ denotes the weight function and $f(x)$ represents the function being approximated.
	The limits $a,b$ and weight function $\omega(x)$ depend on how `J` was constructed. When $J$ arises from the Lanczos
	method on a symmetric matrix $A \in \mathbb{R}^{n \times n}$, the estimated quantity corresponds to a spectral sum:

	$$ \int_{a}^{b} f(x) \psi(x; A, v) dx $$

	where $psi(x)$ is the eigenvector spectral density associated to the pair $(A,v)$:

	$$ \psi(x; A, v) = \sum\limits_{i=1}^n \lvert u_i^T v \rvert^2 \delta(x - \lambda_i), \quad A = U \Lambda U^T $$

	In this sense, by applying $f$ to the nodes $x_i$, the corresponding quadrature rule can approximate any spectral sum.

	Parameters:
		d: array of `n` diagonal elements.
		e: array of `n` or `n-1` off-diagonal elements. See details.
		deg: degree of the quadrature rule to compute.
		quad: method used to compute the rule. Either Golub Welsch or FTTR is supported.
		nodes: output array to store the `n` nodes of the quadrature (optional).
		weights: output array to store the `n` weights of the quadrature (optional).

	Notes:
		To compute the weights of the quadrature, `quad` can be set to either 'golub_welsch' or 'fttr'. The former uses a LAPACK call to
		the method of relatively robust representations (RRR), which builds local LDL decompositions around clusters of eigenvalues,
		while the latter (FTTR) uses the explicit recurrence expression for orthogonal polynomials. Though both require
		$O(\mathrm{deg}^2)$ time to execute, the former requires $O(\mathrm{deg}^2)$ space but is highly accurate, while the latter uses
		only $O(1)$ space at the cost of backward stability. If `deg` is large, `fttr` is preferred for performance, though pilot testing
		should be done to ensure that instability does not cause a large bias in the approximation.
	"""
	deg = len(d) if deg is None else int(min(deg, len(d)))
	e = np.append([0], e) if len(e) == (len(d) - 1) else e
	assert len(d) == len(e) and np.isclose(e[0], 0.0), "Subdiagonal first element 'e[0]' must be close to zero"

	if quad in {"gw", "golub_welsch"}:
		## Golub-Welsch approach: just compute eigen-decomposition from T using QR steps
		theta, ev = eigh_tridiag(d[:deg], e[:deg], **kwargs)
		tau = np.square(ev[0, :])
	elif quad == "fttr":
		## Uses the Forward Three Term Recurrence (FTTR) approach
		theta = eigvalsh_tridiag(d, e, **kwargs)
		tau = np.zeros(len(theta), dtype=theta.dtype)
		fttr(theta, d, e, deg, tau)
	else:
		raise ValueError(f"Invalid quadrature method '{quad}' supplied")
	if nodes is not None and weights is not None:
		assert len(nodes) == deg and len(weights) == deg, "`nodes` and `weights` output arrays must be `deg` in length."
		np.copyto(nodes, theta)
		np.copyto(weights, tau)
	return theta, tau


def spectral_density(
	A: Union[LinearOperator, np.ndarray],
	bins: Union[int, np.ndarray] = 100,
	bw: Union[float, str] = "scott",
	deg: int = 20,
	rtol: float = 0.01,
	verbose: bool = False,
	info: bool = False,
	plot: bool = False,
	**kwargs,
):
	"""Estimates the spectral density of an operator via stochastic Lanczos quadrature.

	Parameters:
		A: real symmetric matrix or linear operator.
		bins: number of domain points to accumulate density
		bw: bandwidth value or rule
		deg: degree of each quadrature approximation
		rtol: relative stopping tolerance
		verbose: whether to report various statistics

	Return:
		(density, bins) = Estimate of the spectral density at domain points 'bins'
	"""
	## First probe info about the spectrum via a single adaptive Krylov expansion
	n = A.shape[0]
	# spec_radius = eigsh(A, k=1, which="LM")
	# spec_radius, info = spectral_radius(A, full_output=True)
	# min_rw = np.min(info["ritz_values"])
	# fun = "identity" if fun is None or (isinstance(fun, str) and fun == "identity") else fun
	# fun = param_callable(fun, kwargs) if isinstance(fun, str) else fun
	# assert isinstance(fun(1.0), Number), "Function must return a real number."

	## Parameterize the kernel
	## Automatic bandwidth determination for "bimodal or multi-modal distributions tend to be oversmoothed."
	N = deg * n
	if bw == "scott":
		h = N ** (-1 / 5)
		h **= 2  # to prevent over-smoothing
	elif bw == "silverman":
		h = (N * 3 / 4) ** (-1 / 7)
		h **= 2  # to prevent over-smoothing
	else:
		assert isinstance(bw, Number), f"Invalid bandwidth estimator '{bw}'; must be 'scott', 'silverman', or float."
		h = bw
	K = lambda u: np.exp(-0.5 * u**2)

	## Prepare the bins for the estimate
	# bins = np.linspace(min_rw, spec_radius, int(bins)) if isinstance(bins, Number) else np.asarray(bins)
	bins = np.linspace(0, 1, int(bins), endpoint=True)
	n_bins = len(bins)
	spectral_density = np.zeros(n_bins)  # accumulate density estimate
	density_residual = np.zeros(n_bins)  # difference in density per iteration
	min_bins = np.inf * np.ones(n_bins)  # min value encountered per bin

	## Begin sampling stochastic quadrature estimates
	rel_change, jj = np.inf, 0
	trace_samples = array("f")
	while rel_change > rtol and jj < A.shape[0]:
		## Acquire a quadrature estimate
		## TODO: The inner sum can likely be vectorized with an einsum or something
		alpha, beta = lanczos(A, deg=deg, **kwargs)
		nodes, weights = lanczos_quadrature(alpha, beta)
		density_residual.fill(0)
		for i, t in enumerate(nodes):
			density_residual += weights[i] * K((bins - t) / h)  # weights[i] * c # Note constant 'c' can be dropped

		# density_residual = np.sum(weights * K((bins[:,np.newaxis] - nodes) / h), axis=1)
		# np.sum(weights * K((bins[:,np.newaxis] - nodes) / h), axis=1)
		# np.sum(weights * (bins[:,np.newaxis] - nodes), axis=1)
		# np.einsum('i,ji,j->j', weights, bins[:, np.newaxis] - nodes, np.ones_like(bins))

		## Maintain a minimum ritz estimate per bin to estimate spectral gap
		bin_ind = np.clip(np.digitize(nodes, bins), 0, n_bins - 1)
		min_bins[bin_ind] = np.minimum(min_bins[bin_ind], nodes)

		## Accumulate the spectral density
		spectral_density += density_residual
		jj += 1
		if jj > 2:
			w1 = (spectral_density - density_residual) / np.sum(spectral_density - density_residual)
			w2 = spectral_density / np.sum(spectral_density)
			rel_change = np.mean(np.abs((w1 - w2) / np.where(w1 > 0, w1, 1.0)))

		## Keep trace of the spectral sum each iteration
		trace_samples.append(np.sum(weights * nodes * n))

	## Normalize such it density approx. integrates to 1
	spectral_density /= np.sum(spectral_density) * np.diff(bins[:2])

	## Plot if requested
	if plot:
		from bokeh.plotting import figure, show

		p = figure(width=700, height=300, title=f"Estimated spectral density (bw = {h:.4f}, n_samples = {jj})")
		p.scatter(bins, spectral_density)
		p.line(bins, spectral_density)
		# y_lb = np.min(spectral_density) - np.ptp(spectral_density) * 0.025
		# p.scatter(ew, , marker='plus', color='red', fill_alpha=0.25, line_width=0, size=6)
		show(p)

	if info:
		info_dict = {
			"trace": np.mean(trace_samples),
			"rtol": rtol,
			"quad_est": trace_samples,
			"bandwidth": h,
			"n_samples": jj,
		}
		return (spectral_density, bins), info_dict
	return (spectral_density, bins)
