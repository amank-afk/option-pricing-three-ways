"""Reproduce every figure and table in the report.

    python scripts/run_experiments.py

Writes PNGs to figures/, CSVs and summary.json to results/. Takes about a minute.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optionpricing import black_scholes as bs
from optionpricing import finite_difference as fd
from optionpricing import trinomial as tri

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Base case used throughout
S0, K, T, r, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY = "#1f4e79", "#d95f02", "#1b9e77", "#7f7f7f"
summary: dict = {}


def timed(f, *a, repeat=3, **k):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = f(*a, **k)
        best = min(best, time.perf_counter() - t0)
    return out, best


# 1. Closed form sanity checks --------------------------------------------
call = bs.price(S0, K, T, r, SIGMA, "call")
put = bs.price(S0, K, T, r, SIGMA, "put")
summary["bs_call"] = float(call)
summary["bs_put"] = float(put)
summary["put_call_parity_error"] = float(call - put - (S0 - K * np.exp(-r * T)))
summary["bs_greeks_call"] = {
    "delta": float(bs.delta(S0, K, T, r, SIGMA)), "gamma": float(bs.gamma(S0, K, T, r, SIGMA)),
    "vega": float(bs.vega(S0, K, T, r, SIGMA)), "theta": float(bs.theta(S0, K, T, r, SIGMA)),
    "rho": float(bs.rho(S0, K, T, r, SIGMA))}

# 2. The explicit scheme IS a trinomial tree --------------------------------
rows = []
for N in (10, 50, 100, 500, 1000):
    for kind, am in (("call", False), ("put", True)):
        v_fd = fd.price_explicit(S0, K, T, r, SIGMA, kind, am, n_steps=N, n_half=N).price
        v_tr = tri.price_trinomial(S0, K, T, r, SIGMA, kind, am, n_steps=N)
        rows.append((N, f"{'American' if am else 'European'} {kind}", v_fd, v_tr, abs(v_fd - v_tr)))
with open(RES / "tree_vs_explicit.csv", "w") as f:
    f.write("n_steps,option,explicit_fd,trinomial_tree,abs_difference\n")
    for row in rows:
        f.write(f"{row[0]},{row[1]},{row[2]:.12f},{row[3]:.12f},{row[4]:.2e}\n")
summary["tree_vs_fd_max_abs_diff"] = float(max(r_[4] for r_ in rows))

# 3. Convergence and cost ----------------------------------------------------
Ns = np.array([25, 50, 100, 200, 400, 800, 1600, 3200])
err_ex, err_tr, err_cn, t_ex, t_cn = [], [], [], [], []
for N in Ns:
    res, t = timed(fd.price_explicit, S0, K, T, r, SIGMA, n_steps=int(N))
    err_ex.append(abs(res.price - call)); t_ex.append(t)
    err_tr.append(abs(tri.price_trinomial(S0, K, T, r, SIGMA, n_steps=int(N), discount="exp") - call))
    res, t = timed(fd.price_theta, S0, K, T, r, SIGMA, n_steps=int(N) // 4 + 1,
                   n_space=int(N), rannacher_steps=2)
    err_cn.append(abs(res.price - call)); t_cn.append(t)
err_ex, err_tr, err_cn = map(np.array, (err_ex, err_tr, err_cn))
slope_ex = np.polyfit(np.log(Ns), np.log(err_ex), 1)[0]
slope_cn = np.polyfit(np.log(Ns), np.log(err_cn), 1)[0]
summary["convergence"] = {"explicit_slope_vs_time_steps": float(slope_ex),
                          "crank_nicolson_slope_vs_space_steps": float(slope_cn)}
with open(RES / "convergence.csv", "w") as f:
    f.write("N,explicit_error,tree_exp_discount_error,cn_error,explicit_seconds,cn_seconds\n")
    for row in zip(Ns, err_ex, err_tr, err_cn, t_ex, t_cn):
        f.write(",".join(f"{v:.6g}" for v in row) + "\n")


def time_to_accuracy(errs, times, tol):
    ok = np.where(errs < tol)[0]
    return float(times[ok[0]]) if len(ok) else None


tol = 1e-3  # a tenth of a cent on a 10.45 option
summary["seconds_to_1e-3_error"] = {
    "explicit": time_to_accuracy(err_ex, np.array(t_ex), tol),
    "crank_nicolson": time_to_accuracy(err_cn, np.array(t_cn), tol)}
# High accuracy: how long does the explicit scheme need to match CN's best result?
target = err_cn[-1]
N_big = int(np.ceil(Ns[-1] * err_ex[-1] / target))  # error ~ 1/N
res_big, t_big = timed(fd.price_explicit, S0, K, T, r, SIGMA, n_steps=N_big, repeat=1)
summary["high_accuracy"] = {
    "target_error": float(target), "cn_space_steps": int(Ns[-1]), "cn_seconds": float(t_cn[-1]),
    "explicit_steps_needed": N_big, "explicit_error": float(abs(res_big.price - call)),
    "explicit_seconds": float(t_big), "speedup": float(t_big / t_cn[-1])}

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].loglog(Ns, err_ex, "o-", c=BLUE, label=f"Explicit FD (slope {slope_ex:.2f})")
ax[0].loglog(Ns, err_tr, "s--", c=ORANGE, ms=4, label="Trinomial tree, exp(-r dt)")
ax[0].loglog(Ns, err_cn, "^-", c=GREEN, label=f"Crank-Nicolson (slope {slope_cn:.2f})")
ax[0].set_xlabel("N (explicit: time steps; CN: space steps)")
ax[0].set_ylabel("|price error|")
ax[0].set_title("Convergence to the Black-Scholes price")
ax[0].legend(fontsize=7)
ax[1].loglog(t_ex, err_ex, "o-", c=BLUE, label="Explicit FD")
ax[1].loglog(t_cn, err_cn, "^-", c=GREEN, label="Crank-Nicolson")
ax[1].loglog([t_big], [abs(res_big.price - call)], "o", mfc="none", c=BLUE,
             label=f"Explicit, N = {N_big:,}")
ax[1].axhline(tol, c=GREY, ls=":", lw=1)
ax[1].text(t_ex[0], tol * 1.3, "0.1 cent", color=GREY, fontsize=7)
ax[1].set_xlabel("CPU time (s)")
ax[1].set_ylabel("|price error|")
ax[1].set_title("Accuracy per unit of computing time")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "convergence.png")
plt.close(fig)

# 4. Stability of the explicit scheme ---------------------------------------
lams = np.array([0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.02, 1.05])
stab = []
for lam in lams:
    res = fd.price_explicit(S0, K, T, r, SIGMA, n_steps=400, lam=lam)
    stab.append((lam, res.p_mid, abs(res.price - call), np.abs(res.V).max()))
with open(RES / "stability.csv", "w") as f:
    f.write("lambda,p_mid,abs_price_error,max_abs_value_on_grid\n")
    for row in stab:
        f.write(",".join(f"{v:.6g}" for v in row) + "\n")
summary["stability"] = {f"{s[0]:g}": {"p_mid": s[1], "abs_error": s[2]} for s in stab}

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for lam, c in ((0.9, BLUE), (1.03, ORANGE)):
    res = fd.price_explicit(S0, K, T, r, SIGMA, n_steps=150, lam=lam)
    m = (res.S > 60) & (res.S < 160)
    ax[0].plot(res.S[m], res.V[m], c=c, lw=1, label=f"$\\lambda$ = {lam}")
Sg = np.linspace(60, 160, 200)
ax[0].plot(Sg, bs.price(Sg, K, T, r, SIGMA), c=GREY, ls="--", lw=1, label="Black-Scholes")
ax[0].set_ylim(-60, 110)
ax[0].set_xlabel("S")
ax[0].set_ylabel("call value")
ax[0].set_title("Explicit scheme, 150 time steps")
ax[0].legend(fontsize=7)
ax[1].semilogy([s[0] for s in stab], [max(s[2], 1e-16) for s in stab], "o-", c=BLUE)
ax[1].axvline(1.0, c=GREY, ls=":", lw=1)
ax[1].set_xlabel("$\\lambda = \\sigma^2\\Delta\\tau/\\Delta x^2$")
ax[1].set_ylabel("|price error| at S = 100")
ax[1].set_title("Error explodes once $p_m < 0$")
fig.tight_layout()
fig.savefig(FIG / "stability.png")
plt.close(fig)

# 5. American puts -------------------------------------------------------------
N_AM = 2000
grid = []
for T_ in (0.25, 1.0, 2.0):
    for K_ in (80.0, 90.0, 100.0, 110.0, 120.0):
        eu = float(bs.price(S0, K_, T_, r, SIGMA, "put"))
        am = tri.price_trinomial(S0, K_, T_, r, SIGMA, "put", True, n_steps=N_AM, discount="exp")
        grid.append((T_, K_, eu, am, am - eu, (am - eu) / eu))
with open(RES / "american_put_premium.csv", "w") as f:
    f.write("T,K,european_put,american_put,early_exercise_premium,premium_pct_of_european\n")
    for row in grid:
        f.write(",".join(f"{v:.6g}" for v in row) + "\n")
summary["american_put_atm_1y"] = next(g[3] for g in grid if g[0] == 1.0 and g[1] == 100.0)
summary["american_premium_atm_1y"] = next(g[4] for g in grid if g[0] == 1.0 and g[1] == 100.0)
summary["american_premium_range_pct"] = [min(g[5] for g in grid) * 100, max(g[5] for g in grid) * 100]

am = fd.price_explicit(S0, K, T, r, SIGMA, "put", True, n_steps=4000, track_boundary=True)
eu = fd.price_explicit(S0, K, T, r, SIGMA, "put", False, n_steps=4000)
summary["exercise_boundary_at_t0"] = float(am.exercise_boundary[-1])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
m = (am.S > 50) & (am.S < 140)
ax[0].plot(am.S[m], am.V[m], c=BLUE, label="American put")
ax[0].plot(eu.S[m], eu.V[m], c=ORANGE, ls="--", label="European put")
ax[0].plot(am.S[m], np.maximum(K - am.S[m], 0), c=GREY, lw=1, label="payoff K - S")
ax[0].set_xlabel("S")
ax[0].set_ylabel("value at t = 0")
ax[0].set_title("Early exercise lifts the put above its payoff")
ax[0].legend(fontsize=7)
t_cal = T - am.tau
ax[1].plot(t_cal, am.exercise_boundary, c=BLUE)
ax[1].fill_between(t_cal, 0, am.exercise_boundary, color=BLUE, alpha=0.12)
ax[1].text(0.35, 80, "exercise now", color=BLUE, fontsize=8)
ax[1].text(0.35, 95, "hold", color=GREY, fontsize=8)
ax[1].set_ylim(70, 102)
ax[1].set_xlabel("calendar time t (years)")
ax[1].set_ylabel("critical price $S^*(t)$")
ax[1].set_title("Optimal exercise boundary, K = 100")
fig.tight_layout()
fig.savefig(FIG / "american_put.png")
plt.close(fig)

# 6. Greeks from the grid, and why Crank-Nicolson needs Rannacher -----------
T_s = 0.25
cn_raw = fd.price_theta(S0, K, T_s, r, SIGMA, n_steps=20, n_space=800, width=6)
cn_ran = fd.price_theta(S0, K, T_s, r, SIGMA, n_steps=20, n_space=800, width=6, rannacher_steps=2)


def grid_gamma(res):
    x = np.log(res.S)
    dx = x[1] - x[0]
    V_x = np.gradient(res.V, dx)
    V_xx = np.gradient(V_x, dx)
    return (V_xx - V_x) / res.S**2


fig, ax = plt.subplots(figsize=(4.6, 3.0))
m = (cn_raw.S > 85) & (cn_raw.S < 115)
ax.plot(cn_raw.S[m], grid_gamma(cn_raw)[m], c=ORANGE, lw=1, label="Crank-Nicolson")
ax.plot(cn_ran.S[m], grid_gamma(cn_ran)[m], c=BLUE, lw=1.5, label="CN + 2 Rannacher steps")
ax.plot(cn_raw.S[m], bs.gamma(cn_raw.S[m], K, T_s, r, SIGMA), c=GREY, ls="--", lw=1,
        label="analytic gamma")
ax.set_xlabel("S")
ax.set_ylabel("gamma")
ax.set_title("Gamma, 3-month call, 20 time steps")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "gamma_rannacher.png")
plt.close(fig)
mm = np.abs(np.log(cn_raw.S / K)) < 0.1
g_exact = bs.gamma(cn_raw.S[mm], K, T_s, r, SIGMA)
summary["gamma_max_abs_error_near_strike"] = {
    "crank_nicolson": float(np.abs(grid_gamma(cn_raw)[mm] - g_exact).max()),
    "with_rannacher": float(np.abs(grid_gamma(cn_ran)[mm] - g_exact).max())}

res = fd.price_explicit(S0, K, T, r, SIGMA, n_steps=3200)
summary["explicit_greeks_N3200"] = {"delta": res.delta, "gamma": res.gamma}

with open(RES / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
