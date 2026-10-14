"""Finite-difference solvers for the Black-Scholes PDE.

Working in the log price x = ln S and time-to-maturity tau = T - t, the
Black-Scholes equation becomes a PDE with constant coefficients

    V_tau = 0.5 sigma^2 V_xx + nu V_x - r V,      nu = r - 0.5 sigma^2,

with the payoff as the initial condition at tau = 0. Two solvers are provided:

* :func:`price_explicit` - the explicit (forward-in-tau) scheme. Each new value
  is a weighted average of three old ones, which is exactly a trinomial tree
  (see :mod:`optionpricing.trinomial`). Stable only when
  lambda = sigma^2 dtau / dx^2 <= 1 (roughly).
* :func:`price_theta` - the theta-scheme family (theta = 0 explicit,
  0.5 Crank-Nicolson, 1 fully implicit), with optional Rannacher start-up
  steps to damp the oscillations Crank-Nicolson produces from the kinked payoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class FDResult:
    """Output of a finite-difference solve."""

    price: float
    delta: float
    gamma: float
    S: np.ndarray            # spot grid at t = 0
    V: np.ndarray            # option values on that grid at t = 0
    p_up: float = np.nan     # explicit-scheme weights (nan for implicit schemes)
    p_mid: float = np.nan
    p_down: float = np.nan
    exercise_boundary: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau: np.ndarray = field(default_factory=lambda: np.empty(0))


def _payoff(S, K, kind):
    if kind == "call":
        return np.maximum(S - K, 0.0)
    if kind == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _boundaries(S_lo, S_hi, K, r, tau, kind, american):
    """Dirichlet values at the edges of the grid at time-to-maturity tau."""
    if kind == "call":
        return 0.0, S_hi - K * np.exp(-r * tau)
    lower = K - S_lo if american else K * np.exp(-r * tau) - S_lo
    return lower, 0.0


def _greeks_from_grid(x, V, j):
    """Delta and gamma at node j from central differences in x = ln S."""
    dx = x[1] - x[0]
    S = np.exp(x[j])
    V_x = (V[j + 1] - V[j - 1]) / (2 * dx)
    V_xx = (V[j + 1] - 2 * V[j] + V[j - 1]) / dx**2
    return V_x / S, (V_xx - V_x) / S**2


def price_explicit(S0: float, K: float, T: float, r: float, sigma: float,
                   kind: str = "call", american: bool = False, n_steps: int = 200,
                   lam: float = 0.5, width: float = 5.0, n_half: int | None = None,
                   track_boundary: bool = False) -> FDResult:
    """Explicit finite-difference price.

    Parameters
    ----------
    n_steps : number of time steps; dtau = T / n_steps.
    lam : lambda = sigma^2 dtau / dx^2, which fixes dx = sigma sqrt(dtau / lam).
        The scheme is stable for lam <= 1 - r dtau (middle weight non-negative).
    width : the grid spans ln S0 +/- width * sigma * sqrt(T).
    n_half : override the number of grid nodes either side of S0. With
        n_half >= n_steps the boundaries can never influence the price at S0.
    track_boundary : for an American put, record the early-exercise boundary.

    S0 always sits exactly on a grid node, so no interpolation is needed.
    """
    dtau = T / n_steps
    dx = sigma * np.sqrt(dtau / lam)
    if n_half is None:
        n_half = int(np.ceil(width * sigma * np.sqrt(T) / dx))
    j = np.arange(-n_half, n_half + 1)
    x = np.log(S0) + j * dx
    S = np.exp(x)

    nu = r - 0.5 * sigma**2
    a = 0.5 * sigma**2 * dtau / dx**2
    b = 0.5 * nu * dtau / dx
    p_up, p_down = a + b, a - b
    p_mid = 1.0 - 2.0 * a - r * dtau

    payoff = _payoff(S, K, kind)
    V = payoff.copy()
    boundary, taus = [], []
    for n in range(1, n_steps + 1):
        tau = n * dtau
        V_new = np.empty_like(V)
        V_new[1:-1] = p_up * V[2:] + p_mid * V[1:-1] + p_down * V[:-2]
        V_new[0], V_new[-1] = _boundaries(S[0], S[-1], K, r, tau, kind, american)
        if american:
            np.maximum(V_new, payoff, out=V_new)
            if track_boundary and kind == "put":
                exercised = (payoff > 0) & (V_new <= payoff + 1e-12)
                boundary.append(S[exercised].max() if exercised.any() else np.nan)
                taus.append(tau)
        V = V_new

    mid = n_half
    d, g = _greeks_from_grid(x, V, mid)
    return FDResult(price=float(V[mid]), delta=float(d), gamma=float(g), S=S, V=V,
                    p_up=p_up, p_mid=p_mid, p_down=p_down,
                    exercise_boundary=np.array(boundary), tau=np.array(taus))


def price_theta(S0: float, K: float, T: float, r: float, sigma: float,
                kind: str = "call", n_steps: int = 100, n_space: int = 200,
                theta: float = 0.5, rannacher_steps: int = 0,
                width: float = 5.0) -> FDResult:
    """European option by the theta-scheme on a uniform log-price grid.

    theta = 0.5 is Crank-Nicolson (second order in time and space, stable for
    any step sizes). ``rannacher_steps`` fully implicit half-steps are taken
    first to smooth the payoff kink. ``n_space`` is the number of intervals;
    it is rounded up to an even number so that S0 lies on the middle node.
    """
    n_space += n_space % 2
    half = n_space // 2
    x = np.log(S0) + np.linspace(-1.0, 1.0, n_space + 1) * width * sigma * np.sqrt(T)
    dx = x[1] - x[0]
    S = np.exp(x)
    nu = r - 0.5 * sigma**2

    # Spatial operator L V = alpha V_{j-1} + beta V_j + gamma V_{j+1}
    alpha = 0.5 * sigma**2 / dx**2 - 0.5 * nu / dx
    beta = -sigma**2 / dx**2 - r
    gamma_ = 0.5 * sigma**2 / dx**2 + 0.5 * nu / dx

    def step(V, tau_new, dtau, th):
        m = len(V) - 2
        # Right-hand side: (I + (1-th) dtau L) V on interior nodes
        rhs = V[1:-1] + (1 - th) * dtau * (alpha * V[:-2] + beta * V[1:-1] + gamma_ * V[2:])
        lo, hi = _boundaries(S[0], S[-1], K, r, tau_new, kind, american=False)
        rhs[0] += th * dtau * alpha * lo
        rhs[-1] += th * dtau * gamma_ * hi
        ab = np.zeros((3, m))
        ab[0, 1:] = -th * dtau * gamma_
        ab[1, :] = 1 - th * dtau * beta
        ab[2, :-1] = -th * dtau * alpha
        V_new = np.empty_like(V)
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)
        V_new[0], V_new[-1] = lo, hi
        return V_new

    dtau = T / n_steps
    V = _payoff(S, K, kind)
    tau = 0.0
    for n in range(n_steps):
        if n < rannacher_steps:   # two implicit half-steps replace one CN step
            for _ in range(2):
                tau += dtau / 2
                V = step(V, tau, dtau / 2, 1.0)
        else:
            tau += dtau
            V = step(V, tau, dtau, theta)

    d, g = _greeks_from_grid(x, V, half)
    return FDResult(price=float(V[half]), delta=float(d), gamma=float(g), S=S, V=V)
