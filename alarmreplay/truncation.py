#!/usr/bin/env python3
"""Right-truncated data: the reverse product-limit estimator, the conditional Kendall
statistic that tests quasi-independence, and the structural transformation that addresses
a failure of it by fitting L = b*D + eps and estimating the law of the residual.
"""
import numpy as np

def rev_product_limit(x, t, min_risk=5):
    x = np.asarray(x, float); t = np.asarray(t, float)
    if x.size == 0: return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    u, d = np.unique(x, return_counts=True); d = d.astype(float)
    xs, ts = np.sort(x), np.sort(t)
    Rk = (np.searchsorted(xs, u, side="right") - np.searchsorted(ts, u, side="left")).astype(float)
    ok = (Rk >= min_risk) & (Rk > d)
    if not ok.any(): return np.array([float(u.min()), float(u.max())]), np.array([0.0, 1.0])
    hi = int(np.flatnonzero(ok).max())
    uu, dd, RR = u[:hi+1], d[:hi+1], Rk[:hi+1]
    fac = np.clip(1.0 - np.divide(dd, RR, out=np.zeros_like(dd), where=RR > 0), 1e-12, 1.0)
    F = np.cumprod(fac[::-1])[::-1]
    return uu, np.clip(np.r_[F[1:], 1.0], 0.0, 1.0)

def tau_pairs(n, K, rng):
    """K random index pairs: a subsampled U-statistic, O(K) instead of O(n^2) per evaluation."""
    i = rng.integers(0, n, size=K); j = rng.integers(0, n, size=K)
    m = i != j
    return i[m], j[m]

def tau_c(x, t, ii, jj):
    """Conditional Kendall tau over comparable pairs: max(x_i,x_j) <= min(t_i,t_j)."""
    xi, xj, ti, tj = x[ii], x[jj], t[ii], t[jj]
    comp = np.maximum(xi, xj) <= np.minimum(ti, tj)
    dx = np.sign(xi - xj); dt = np.sign(ti - tj)
    sel = comp & (dx != 0) & (dt != 0)
    if sel.sum() < 30: return np.nan
    return float((dx * dt)[sel].mean())

def fit_b(Th, D, rng, K=120000, lo=-0.6, hi=0.95, iters=22, m=0.0):
    """Solve tau_c(b) = 0 by bisection. tau_c is decreasing in b: subtracting more of D removes
    more positive dependence. m is the truncation margin (see module docstring)."""
    ii, jj = tau_pairs(len(Th), K, rng)
    def f(b):
        return tau_c(Th - b*D, (1.0 - b)*D - m, ii, jj)
    flo, fhi = f(lo), f(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi): return 0.0, np.nan
    if flo < 0: return lo, flo              # already negative at the left end
    if fhi > 0: return hi, fhi
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        fm = f(mid)
        if not np.isfinite(fm): break
        if fm > 0: lo = mid
        else: hi = mid
    b = 0.5*(lo + hi)
    return b, f(b)

def integrate_step_cdf(c, u, F):
    """EXACT value of int_{-inf}^{c} F(s) ds for the right-continuous step CDF with jumps at u to F.

    A trapezoid rule on a linear grid is not safe here: the identified region can extend to hundreds of
    seconds while the durations that matter are tens, so a fixed grid loses all resolution where psi is
    evaluated. Integrating the step function in closed form removes the grid entirely.
    """
    c = np.atleast_1d(np.asarray(c, float))
    u = np.asarray(u, float); F = np.asarray(F, float)
    w = np.diff(u)                                     # width of each step [u_k, u_{k+1})
    full = np.concatenate([[0.0], np.cumsum(F[:-1]*w)])  # integral up to u_k
    k = np.clip(np.searchsorted(u, c, side="right") - 1, -1, len(u)-1)
    out = np.zeros_like(c)
    inside = k >= 0
    kk = k[inside]
    out[inside] = full[kk] + F[kk]*(c[inside] - u[kk])   # partial last step; F is 0 below u[0]
    return np.clip(out, 0.0, None)

def psi_from_F(dgrid, u, F):
    """psi(d) = int_0^d F, evaluated exactly on a step estimate."""
    return integrate_step_cdf(np.asarray(dgrid, float), u, F)

def psi_transformed(dgrid, Th, D, b, m=0.0):
    """psi_hat(d) = int_{-inf}^{(1-b)d} F_eps, with F_eps from Lynden-Bell on (eps, (1-b)D - m).
    At b = 0 this reduces exactly to the pooled estimator with truncation variable D - m."""
    eps = Th - b*D; V = (1.0 - b)*D - m
    u, F = rev_product_limit(eps, V)
    return integrate_step_cdf((1.0 - b)*np.asarray(dgrid, float), u, F)

# --- lattice data -------------------------------------------------------------------------------
# Offsets are recorded on a grid (1 s here, 2 s for most values), so Theta carries heavy ties. A rank
# statistic is degenerate there: at b=0 tied pairs are dropped, while any b != 0 breaks those ties
# SYSTEMATICALLY by D, driving tau_c to +-1 and putting a jump discontinuity at b=0. Randomised
# de-rounding -- adding U(-h/2, h/2) to Theta, with h the grid spacing -- recovers a sample from the
# underlying continuous law and breaks ties at random instead; tau_c(b) is then smooth and monotone.
#
# Two things must be done exactly. (1) D is NOT jittered: durations are exact counts of samples, and
# jittering them blurs the observation rule by up to a grid step, which biases the statistic at the
# truth (in simulation tau_c(b0) = +0.11 instead of 0). (2) The truncation variable keeps the rule
# coherent: Theta <= D - m on the grid is EXACTLY Theta~ < D - m + h/2 for the jittered Theta~ when
# Theta is on a grid of spacing h, so V(b) = (1-b)D - m + h/2. With these two, tau_c(b0) is zero to
# sampling error on lattice data (simulation: +0.002 at the truth) and smooth in b.
# De-rounding is used ONLY for the estimating equation; the product-limit step handles ties natively
# through the multiplicity term and is applied to the recorded values with V = D - m.

def tau_c_derounded(Th, D, b, rng, J=5, K=120000, h=1.0, m=0.0):
    vals = []
    for _ in range(J):
        Tj = Th + rng.uniform(-h/2, h/2, Th.size)
        ii, jj = tau_pairs(Th.size, K, rng)
        v = tau_c(Tj - b*D, (1.0 - b)*D - m + h/2, ii, jj)
        if np.isfinite(v): vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def fit_b_derounded(Th, D, rng, J=5, K=120000, lo=-0.6, hi=0.95, iters=18, h=1.0, m=0.0):
    """Solve tau_c(b) = 0 on de-rounded data. Returns (b_hat, tau_c at b_hat)."""
    f = lambda b: tau_c_derounded(Th, D, b, rng, J, K, h, m)
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)): return 0.0, np.nan
    if flo < 0: return lo, flo
    if fhi > 0: return hi, fhi
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        fm = f(mid)
        if not np.isfinite(fm): break
        if fm > 0: lo = mid
        else: hi = mid
    b = 0.5*(lo + hi)
    return b, f(b)

# --- the observation mechanism of the matcher: L is truncated, E is not --------------------------
# A pair is recorded when the annunciation run overlaps the excursion. With D = e - s + 2, L = As - s and
# E = Ae - e, positive overlap is L <= D - 3 (the onset lag is RIGHT-TRUNCATED by the duration less the
# grid margin) together with E >= -(D - 3), which is non-binding except for a few very short excursions.
# Theta = L - E is therefore not itself the truncated variable: L is, and E is observed freely. The
# estimator below truncation-corrects F_L (with the transformation L = bD + eps for dependence), takes E
# from its empirical conditional law given the duration stratum, and forms
#     psi(d) = E[(d - L + E)_+ | D = d] = E_E[ int_{-inf}^{(1-b)d + E} F_eps ].
# Both the estimating equation and the product-limit run on de-rounded L (jitter U(-h/2, h/2), D exact,
# truncation variable (1-b)D - m + h/2), and the product-limit integral carries the exact boundary
# correction (h/8) x (jump mass in the cell at the upper limit), which is what the jitter otherwise adds.

def psi_LE(dgrid, L, E, D, b, rng, m=3.0, h=1.0, J=5, cuts=None, bw=0.30):
    """psi_hat(d) = E_{E|D=d}[ int_{-inf}^{(1-b)d+E} F_eps ], with F_eps the de-rounded, boundary-corrected
    reverse product-limit of the transformed lag, and the conditional law of the extension estimated by a
    Gaussian kernel on log-duration (bandwidth bw in log units; bw=0.30 is a +-35% window). cuts=(...)
    replaces the kernel by hard duration strata (kept for comparison); cuts=(0, inf) pools E entirely."""
    dgrid = np.asarray(dgrid, float); L = np.asarray(L, float); E = np.asarray(E, float); D = np.asarray(D, float)
    if cuts is not None:
        cuts = np.asarray(cuts, float)
        strat = np.clip(np.searchsorted(cuts, D, side="right") - 1, 0, len(cuts) - 2)
        dstrat = np.clip(np.searchsorted(cuts, dgrid, side="right") - 1, 0, len(cuts) - 2)
        W = np.zeros((dgrid.size, D.size))
        for i in range(dgrid.size): W[i, strat == dstrat[i]] = 1.0
    else:
        ld = np.log(np.maximum(D, 1.0)); lg = np.log(np.maximum(dgrid, 1.0))
        W = np.exp(-0.5*((lg[:, None] - ld[None, :])/bw)**2)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)
    acc = np.zeros(dgrid.size)
    for _ in range(J):
        Lj = L + rng.uniform(-h/2, h/2, L.size)
        u, F = rev_product_limit(Lj - b*D, (1.0 - b)*D - m + h/2)
        def Fat(c):
            k = np.clip(np.searchsorted(u, c, side="right") - 1, 0, len(u) - 1)
            return np.where(c < u[0], 0.0, F[k])
        for i, d in enumerate(dgrid):
            c = (1.0 - b)*d + E
            val = integrate_step_cdf(c, u, F) - (h/8.0)*(Fat(c + h/2) - Fat(c - h/2))
            acc[i] += float(W[i] @ val)
    return np.clip(acc / J, 0.0, None)
