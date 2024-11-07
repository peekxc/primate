# %% Imports
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from primate.quadrature import lanczos_quadrature, spectral_density
from primate.stochastic import symmetric

output_notebook()

## trace == 1
A = symmetric(n=5, ew=[0, 0.1, 0.2, 0.3, 0.4], pd=False)

## Q1: can we infer the nullspace from a degree-r Krylov expansion?
(dens, bins), info = spectral_density(A, bw=0.01, plot=True, rtol=0.0001, deg=3, orth=0, info=True)

## Q2: Is the error associated with the Krylov expansion 'uniform' in any sense?
density = np.zeros(100)
for i in range(1500):
	(dens, bins), info = spectral_density(A, bw=0.01, plot=False, deg=3, rtol=0.0001)
	density += dens

## Q2a: given a density that 'looks nice', does the area under the curve line up with the rank?
np.sum(dens[bins <= 0.05]) / np.sum(dens)  # about 20.9%
np.sum(dens[bins > 0.05]) / np.sum(dens)  # about 80%

## Ans: while the density does seem to converge to something, its not centered at the eigenvalues
p = figure(width=350, height=150)
p.line(bins, density)
show(p)

# np.linalg.eigh(A)[0]

# %%
from bokeh.layouts import column, row
from bokeh.models import Band, ColumnDataSource
from landmark import landmarks
from primate.lanczos import lanczos, OrthogonalPolynomialBasis
from primate.stochastic import symmetric
from primate.quadrature import spectral_density

rng = np.random.default_rng(1234)
xx = rng.uniform(size=35, low=0, high=1)
ew = np.sort(xx[landmarks(xx[:, np.newaxis], 25)])

A = symmetric(len(ew), ew=ew)
ew, ev = np.linalg.eigh(A)

v = rng.uniform(size=A.shape[0])
v /= np.linalg.norm(v)
a, b = lanczos(A, v=v, deg=20)
nodes, weights = lanczos_quadrature(a, b)


fx, x = spectral_density(A, bins=1500)

## Cumulative spectral density
x = np.linspace(0, 1, 5000)
csm = lambda x: np.searchsorted(ew, x) * (1 / len(ew))

best_fit = False

for deg in [2, 6, 10, 14, 18, 22]:
	p = figure(
		width=350,
		height=250,
		title="Cumulative spectral density",
		x_axis_label="Spectrum",
		y_axis_label=r"$$\mathbf{1}(\lambda \leq x)$$",
	)
	p.title.align = "center"
	p.line(x, csm(x))
	p.scatter(ew, 0, size=7.5, color="red", marker="x", legend_label="Eigenvalues")
	p.varea_step(x=np.append(ew, 1.0), y1=np.zeros(len(ew) + 1), y2=np.append(csm(ew), 1.0), fill_alpha=0.15)
	p.legend.location = "top_left"
	p.legend.margin = 5
	p.legend.padding = 2
	p.toolbar_location = None
	# show(p)

	## Quadrature approximation
	nodes, weights = lanczos_quadrature(*lanczos(A, deg=deg))
	quad_est = A.shape[0] * np.sum(nodes * weights)

	## Best fitting polynomial
	if best_fit:
		p_coeff, resid, rk, sv, rcond = np.polyfit(x, csm(x), deg=2 * deg - 1, full=True)
		# p_coeff, resid, rk, sv, rcond = np.polyfit(ew, csm(ew), deg=7, full=True)
		f = np.poly1d(p_coeff)
		y = f(x)
	else:
		f = OrthogonalPolynomialBasis(A, deg=deg)
		y = f.fit(x, csm(x))

	q = figure(
		width=350,
		height=250,
		title=f"Gaussian quadrature estimate (d={deg})",
		x_axis_label="Spectrum",
		y_axis_label=r"$$\mathcal{P}_{2d-1}(\lambda \leq x)$$",
	)
	q.line(x, csm(x))
	q.varea_step(x=np.append(ew, 1.0), y1=np.zeros(len(ew) + 1), y2=np.append(csm(ew), 1.0), fill_alpha=0.15)
	q.line(x, y)
	# q.scatter(nodes, csm(nodes), size=7.5, line_color="black", fill_color="white")
	# show(q)

	## Get the above and below shapes
	from itertools import pairwise

	import shapely as sp
	from scipy.optimize import brentq
	from shapely import LineString, Polygon

	def find_intersections(f, g, a, b, num_points=1000):
		def h(x):
			return f(x) - g(x)

		# Create a linspace to scan for sign changes
		x_vals = np.linspace(a, b, num_points)
		y_vals = h(x_vals)

		# Find where h(x) changes sign, indicating a root
		roots = []
		for i in range(len(x_vals) - 1):
			if np.sign(y_vals[i]) != np.sign(y_vals[i + 1]):
				# Brent's method to find the root in this sub-interval
				root = brentq(h, x_vals[i], x_vals[i + 1])
				roots.append(root)
		return roots

	crit_points = np.array(find_intersections(csm, f, 0, 1, 10000))
	crit_points = np.unique(np.sort(np.concatenate([crit_points, ew])))
	crit_points = np.append(crit_points[np.flatnonzero(np.diff(crit_points) > 1e-10)], crit_points[-1])

	patches, patch_colors = [], []
	for x1, x2 in pairwise(crit_points):
		y1, y2 = min(f(x1 + 1e-9), csm(x1 + 1e-9)), max(f(x2), csm(x2))
		f_poly = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
		f_line = LineString(np.c_[np.linspace(x1, x2, 15), f(np.linspace(x1, x2, 15))])
		polys_ud = sp.ops.split(f_poly, f_line)

		if len(polys_ud.geoms) == 2:
			st_xy = np.array(list(polys_ud.geoms[0].exterior.coords))
			sb_xy = np.array(list(polys_ud.geoms[1].exterior.coords))
			st_center = np.array(list(polys_ud.geoms[0].centroid.coords)[0])
			sb_center = np.array(list(polys_ud.geoms[1].centroid.coords)[0])
			if sb_center[1] > st_center[1]:
				st_xy, sb_xy = sb_xy, st_xy
			if f(x1 + 1e-4) > csm(x1 + 1e-4):
				patches.append(sb_xy)
				patch_colors.append("red")
			else:
				patches.append(st_xy)
				patch_colors.append("green")
	patch_colors = np.array(patch_colors)
	for col in ["green", "red"]:
		xs = [patch[:, 0] for patch, c in zip(patches, patch_colors) if c == col]
		ys = [patch[:, 1] for patch, c in zip(patches, patch_colors) if c == col]
		label = "Positive error" if col == "green" else "Negative error"
		q.patches(xs, ys, color=col, fill_alpha=0.50, line_width=0.0, legend_label=label)
	q.legend.location = "top_left"
	q.title.align = "center"
	q.legend.margin = 5
	q.legend.padding = 2
	q.toolbar_location = None
	# show(q)

	p.multi_line(xs=[(n, n) for n in nodes], ys=[(0, n) for n in csm(nodes)], line_color="black")
	p.scatter(
		nodes, csm(nodes), size=np.sqrt(weights) * 18, line_color="black", fill_color="white", legend_label="Quad. nodes"
	)
	show(row(p, q))


# %% Orthogonal polynomials
from primate import fttr
from primate.lanczos import lanczos
from primate.tridiag import eigh_tridiag


a, b = lanczos(A, deg=10)
theta, rv = eigh_tridiag(a, b)
tau = np.square(rv[0, :])
k = len(a)
z = np.zeros(len(a), dtype=A.dtype)
mu_0 = np.sum(np.abs(theta[:k]))
mu_sqrt_rec = 1.0 / np.sqrt(mu_0)


dom = np.linspace(0, 1, 250)
P = []
for x in dom:
	fttr.ortho_poly(x, mu_sqrt_rec, a, b, z, k)
	P.append(z.copy())
P = np.array(P)

from map2color import map2hex

colors = map2hex(np.arange(6), "turbo")
p = figure(width=350, height=250)
for d, l_col in zip(range(6), colors):
	# p.scatter(dom, P[:, d], size=2)
	p.line(dom, P[:, d], color=l_col)
show(p)

## Use least squares fitting for coefficients of orthogonal polynomial basis
c = np.linalg.solve(P.T @ P, P.T @ csm(dom))
p.line(dom, P @ c)
show(p)


basis = OrthogonalPolynomialBasis(A, 20)
basis.fit(dom, csm(dom))
basis(np.linspace(0, 1, 10))


# %% Cumulative empirical spectral measure
# np.power(ev.T @ v, 2)
from primate.lanczos import rayleigh_ritz

rw, Y, Q = rayleigh_ritz(A, deg=20, return_eigenvectors=True, v0=v, return_basis=True)
cw = np.square((Q @ Y).T @ v)

x = np.linspace(0, 1, 5000)
cesm = lambda x: np.array([np.sum(cw[:xi]) for xi in np.searchsorted(rw, x)])


p = figure(
	width=350,
	height=250,
	title="Cumulative empirical spectral density",
	x_axis_label="Spectrum",
	y_axis_label=r"$$\mathbf{1}(\lambda \leq x)$$",
)
p.title.align = "center"
p.line(x, cesm(x))
p.scatter(ew, 0, size=7.5, color="red", marker="x", legend_label="Eigenvalues")
p.varea_step(x=np.append(rw, 1.0), y1=np.zeros(len(rw) + 1), y2=np.append(cesm(rw), 1.0), fill_alpha=0.15)
p.legend.location = "top_left"
p.legend.margin = 5
p.legend.padding = 2
p.toolbar_location = None
show(p)

# TODO: https://stackoverflow.com/questions/46564099/what-are-the-steps-to-create-a-radar-chart-in-bokeh-python

# from shapely import LineString, Polygon, MultiPolygon
# from itertools import pairwise

# curve = LineString(np.c_[x, f(x)])
# xr = np.append(ew, 1.0)
# yr = np.append(csm(ew), 1.0)
# rectangles = [Polygon([(xl, 0), (xr, 0), (xr, yt), (xl, yt)]) for ((xl, xr), yt) in zip(pairwise(xr), yr)]
# varea = MultiPolygon(rectangles)
# varea = varea.union(varea)
