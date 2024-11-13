import numpy as np
from primate.stats import CentralLimitCriterion, ControlVariableEstimator
from bokeh.plotting import show, figure
from bokeh.io import output_notebook

output_notebook()

# %%
rng = np.random.default_rng(1235)
x = np.zeros(5)
a = np.array([1, 2, 3, 1, 2])
U = rng.uniform(low=0, high=1, size=(1000, 5))
h = lambda x: np.min([x[0] + x[3], x[0] + x[2] + x[4], x[1] + x[2] + x[3], x[1] + x[4]])

## Ground truth
mu = 1339 / 1440

## Crude estimator
y = np.apply_along_axis(h, 1, U * a)
mu_crude = np.mean(y)

## Antithetic estimator
U1 = np.apply_along_axis(h, 1, U[:500] * a)
U2 = np.apply_along_axis(h, 1, (1 - U[:500]) * a)
mu_antithetic = np.sum((U1 + U2) / len(U))

## Control variable
mu_cv = 15 / 16
h_cv = lambda x: np.min([x[0] + x[3], x[1] + x[4]])
y_cv = np.apply_along_axis(h_cv, 1, U * a)
C_cv = np.cov(np.c_[y, y_cv], rowvar=False)
a_cv = C_cv[0, 1] / C_cv[1, 1]
mu_cv_est = np.mean(y - a_cv * (y_cv - mu_cv))
# np.mean(y) - a_cv * (np.mean(y_cv) - mu_cv)
# print(f"Control variable estimator: {mu_cv_est:.8f}")

## See the sample mean differences
print(f"True:        {mu:.6f}")
print(f"Crude:       {mu_crude:.6f}  (error={np.abs(mu_crude - mu):.6f})")
print(f"Antithetic:  {mu_crude:.6f}  (error={np.abs(mu_antithetic - mu):.6f})")
print(f"Control var: {mu_cv_est:.6f}  (error={np.abs(mu_cv_est - mu):.6f})")

# %%  Using the estimator API
sc = CentralLimitCriterion(confidence=0.95, atol=0.1)
sc.update(y)
print(sc)
sc.estimate

## Control variable method
sc = ControlVariableEstimator(mu_cv)
sc.update(y, y_cv)
print(sc)
sc.estimate


# %% Confidence bands
sc = CentralLimitCriterion(confidence=0.95, atol=0.1)
conf_band = []
for yi in y:
	sc.update(np.array([yi]))
	conf_band.append(sc.estimate + sc.margin_of_error * np.array([-1, +1]))
conf_band = np.array(conf_band)

p = figure(width=250, height=150)
p.line(np.arange(len(y))[1:], conf_band[1:, 0])
p.line(np.arange(len(y))[1:], conf_band[1:, 1])
show(p)

# %%
sc = ControlVariableEstimator(mu_cv)
conf_band = []
cv_var2 = []
for yi, yci in zip(y, y_cv):
	sc.update(np.array([yi]), np.array([yci]))
	conf_band.append(sc.estimate + sc.margin_of_error * np.array([-1, +1]))
	cv_var2.append((1.0 / sc.n_samples) * (1.0 - sc.r_sq) * sc.cov.covariance()[0, 0])
conf_band = np.array(conf_band)

p = figure(width=250, height=150)
p.line(np.arange(len(y))[1:], conf_band[1:, 0], color="red")
p.line(np.arange(len(y))[1:], conf_band[1:, 1], color="red")

show(p)


# cv_var = np.array([((1 - np.corrcoef(y[:j], y_cv[:j])[0, 1] ** 2) * np.var(y[:j])) / len(y[:j]) for j in range(len(y))])

# cv_est = np.cumsum((y - sc.alpha * (y_cv - sc.ecv))) / (np.arange(len(y)) + 1)
# cv_var = np.array([np.var(cv_est[:j]) for j in range(len(cv_est))])

# moe = sc.z * np.sqrt(cv_var / (np.arange(len(y)) + 1))

# p = figure(width=250, height=150)
# p.line(np.arange(len(y))[1:], cv_est[1:] + moe[1:], color="red")
# p.line(np.arange(len(y))[1:], cv_est[1:] - moe[1:], color="red")

# show(p)


# show(est.plot(samples))

## Control Variable
