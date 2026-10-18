"""Power simulation for the V2-GRPO vs V0-GRPO primary endpoint (evidence/prereg/power_analysis_amendment.md).

Endpoint: overall live-verified recovery rate on DEV (400 prompts = 6 variant clusters 50/35/86/63/81/85; families A1=85,
A2=86, B=229), paired by prompt, G samples per prompt per arm, S training seeds per arm. No model output is used: this is
a generative simulation. SEED fixed below; output evidence/prereg/power_analysis_results.json.

Latent model on the logit scale (total prompt-level latent variance TAU2 = 1):
  logit p[arm, i, s] = b0 + u_c(i) + v_i + arm*(delta + a_c(i)) + r[arm, s]
  u_c ~ N(0, rho*TAU2) cluster baseline   v_i ~ N(0, (1-rho)*TAU2) prompt difficulty, SHARED by the two arms (pairing)
  a_c ~ N(0, rho*TAU2) cluster-specific treatment effect   r ~ N(0, SIGMA_RUN^2) training-run shift per (arm, seed)
  y ~ Bernoulli(p), G draws per (arm, prompt, seed).  b0, delta calibrated so the marginal rates are base and base+D.
Analyses (reject = two-sided 95% CI excludes 0 and the estimate is positive):
  variant6   cluster-robust t, 6 variant clusters (df 5)       family3  cluster-robust t, 3 families (df 2)
  prompt     ordinary paired t over prompts (ignores clustering and training-seed noise: anticonservative reference)
  seed       paired t over per-seed differences (df S-1; captures training-run noise; S>=2 only)
"""
import json
import sys
import time

import numpy as np

T975 = {1: 12.706, 2: 4.303, 4: 2.776, 5: 2.571, 399: 1.966}   # two-sided 95% t critical values (scipy unavailable)


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

SEED = 20261018
TAU2, SIGMA_RUN = 1.0, 0.4
SIZES = [50, 35, 86, 63, 81, 85]            # A1_cfg, A1_src, A2, B_SKIP, B_DUP, B_OK
FAM = [0, 0, 1, 2, 2, 2]
N = sum(SIZES)
CL = np.repeat(np.arange(6), SIZES)
FM = np.repeat(FAM, SIZES)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
rng = np.random.default_rng(SEED)


def calibrate(base, D, rho):
    """b0 and delta so that E[p0]=base, E[p1]=base+D, integrating over all latent effects (large fixed draw)."""
    r = np.random.default_rng(1)
    M = 400000
    lat = r.normal(0, np.sqrt(TAU2), M)                  # u+v combined has variance TAU2 regardless of rho
    a = r.normal(0, np.sqrt(rho * TAU2), M)
    def mean(b0, d1):
        return expit(b0 + lat).mean(), expit(b0 + lat + d1 + a).mean()
    lo, hi = -12, 12
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(mid, 0)[0] < base else (lo, mid)
    b0 = (lo + hi) / 2
    if D == 0:
        return b0, 0.0
    lo, hi = 0, 12
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(b0, mid)[1] < base + D else (lo, mid)
    return b0, (lo + hi) / 2


def crse(d, cl, ncl):
    """cluster-robust t of the mean of d. d: (reps, N)."""
    est = d.mean(1)
    e = d - est[:, None]
    tot = np.stack([e[:, cl == c].sum(1) for c in range(ncl)], 1)
    se = np.sqrt(ncl / (ncl - 1) * (tot ** 2).sum(1)) / d.shape[1]
    return est, est / se, ncl - 1


def simulate(base, D, rho, S, G, reps):
    b0, dl = calibrate(base, D, rho)
    u = rng.normal(0, np.sqrt(rho * TAU2), (reps, 6))[:, CL]
    v = rng.normal(0, np.sqrt((1 - rho) * TAU2), (reps, N))
    a = rng.normal(0, np.sqrt(rho * TAU2), (reps, 6))[:, CL]
    run = rng.normal(0, SIGMA_RUN, (reps, 2, S))
    ys = []
    for arm in (0, 1):
        eta = (b0 + u + v + arm * (dl + a))[:, None, :] + run[:, arm, :, None]      # reps, S, N
        p = expit(eta)
        ys.append(rng.binomial(G, p) / G)                                             # mean of G draws
    diff = ys[1] - ys[0]                                                              # reps, S, N
    d = diff.mean(1)                                                                  # average over seeds
    out = {}
    est, t, df = crse(d, CL, 6);  out["variant6"] = (est > 0) & (t > T975[df])
    est, t, df = crse(d, FM, 3);  out["family3"] = (est > 0) & (t > T975[df])
    se = d.std(1, ddof=1) / np.sqrt(N); t = d.mean(1) / se
    out["prompt"] = (d.mean(1) > 0) & (t > T975[N - 1])
    if S >= 2:
        ds = diff.mean(2)                                                              # reps, S
        t = ds.mean(1) / (ds.std(1, ddof=1) / np.sqrt(S))
        out["seed"] = (ds.mean(1) > 0) & (t > T975[S - 1])
    return {k: float(x.mean()) for k, x in out.items()}


def mdd(base, rho, S, G, reps, grid):
    pw = {k: [] for k in ("variant6", "family3", "prompt") + (("seed",) if S >= 2 else ())}
    for D in grid:
        r = simulate(base, D, rho, S, G, reps)
        for k in pw:
            pw[k].append(r[k])
    res = {}
    for k, ps in pw.items():
        ps = np.array(ps)
        idx = np.where(ps >= 0.8)[0]
        if len(idx) == 0:
            res[k] = None
        else:
            j = idx[0]
            res[k] = float(grid[j]) if j == 0 else float(np.interp(0.8, [ps[j - 1], ps[j]], [grid[j - 1], grid[j]]))
    return res, {k: [round(float(x), 3) for x in v] for k, v in pw.items()}


if __name__ == "__main__":
    t0 = time.time()
    grid = np.round(np.arange(0.02, 0.61, 0.02), 2)
    out = dict(seed=SEED, reps=REPS, tau2=TAU2, sigma_run=SIGMA_RUN, sizes=SIZES, grid=grid.tolist(), cells=[])
    for base in (0.1, 0.3, 0.5, 0.678):
        for rho in (0.1, 0.3, 0.5):
            for S in (1, 3, 5):
                for G in ((4, 8) if rho == 0.3 else (4,)):
                    m, pw = mdd(base, rho, S, G, REPS, grid)
                    null = simulate(base, 0.0, rho, S, G, REPS * 2)
                    out["cells"].append(dict(base=base, rho=rho, seeds=S, G=G, mdd_80=m, type1_at_zero=null, power_curve=pw))
                    print(base, rho, S, G, m, "type1", {k: round(v, 3) for k, v in null.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("evidence/prereg/power_analysis_results.json", "w"), indent=1)
