import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def safeln(x):
    res = 0
    try:
        res = np.log(x)
    except:
        res = -1500
    return res


@np.vectorize
def vdensity(x, l, s):
    return norm.pdf(x, loc=l, scale=s)


def negloglik1(x, retdf):
    density = norm.pdf(retdf, loc=x[0], scale=x[1])
    loglik = np.sum(np.vectorize(safeln)(density))
    return -loglik


def estimate_loc_scale(retdf):
    x0 = [0.0, 0.01]  # loc,scale

    res = minimize(negloglik1, x0, method="Nelder-Mead", tol=1e-6, args=(retdf))
    return res.x[0], res.x[1], -res.fun


def negloglik2(x, retdf, density, loc_est, scl_est):
    n = density.shape[0]
    adjloc = x[0]
    adjscl = x[1]
    loc = [0] * n
    loc[0] = loc_est
    scl = [0] * n
    scl[0] = scl_est
    gradloc = [0] * n
    gradscl = [0] * n
    for i in range(n):
        gradloc[i] = density[i] * (retdf[i] - loc[i]) / scl[i] ** 2
        gradscl[i] = (density[i] * (retdf[i] - loc[i]) / scl[i] ** 3) - (
            density[i] / scl[i]
        )
        if i != n - 1:
            loc[i + 1] = loc[i] + adjloc * gradloc[i]
            scl[i + 1] = scl[i] + adjscl * gradscl[i]
    new_density = vdensity(retdf, loc, scl)
    loglik = np.sum(np.vectorize(safeln)(new_density))
    return -loglik


def gasmodel(retdf):
    loc_est, scl_est, baseloglik = estimate_loc_scale(retdf)
    density = norm.pdf(retdf, loc=loc_est, scale=scl_est)

    res = minimize(
        negloglik2,
        [0.0, 0.0],
        method="Nelder-Mead",
        tol=1e-6,
        args=(retdf, density, loc_est, scl_est),
    )
    adj_loc = res.x[0]
    adj_scl = res.x[1]

    # calculate parameters over time
    n = density.shape[0]
    loc = [0] * n
    loc[0] = loc_est
    scl = [0] * n
    scl[0] = scl_est
    gradloc = [0] * n
    gradscl = [0] * n
    newdensity = density.copy()
    for i in range(n):
        gradloc[i] = (newdensity[i] * (retdf[i] - loc[i]) / scl[i] ** 2) / 10**6
        gradscl[i] = (
            (newdensity[i] * (retdf[i] - loc[i]) / scl[i] ** 3)
            - (newdensity[i] / scl[i])
        ) / 10**6
        if i != n - 1:
            loc[i + 1] = loc[i] + adj_loc * gradloc[i]
            scl[i + 1] = scl[i] + adj_scl * gradscl[i]

    return pd.DataFrame({"location": loc, "scale": scl}, index=retdf.index).iloc[1:, :]
