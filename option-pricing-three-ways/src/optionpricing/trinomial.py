"""A recombining trinomial tree on the log price.

Over one step of length dt the log price moves up by dx, stays put, or moves
down by dx with probabilities q_u, q_m, q_d chosen so that the first two
moments of the step match those of Brownian motion with drift nu = r - sigma^2/2:

    E[dx-step]   = (q_u - q_d) dx        = nu dt       (to leading order)
    E[dx-step^2] = (q_u + q_d) dx^2      = sigma^2 dt   (to leading order)

With the explicit finite-difference weights p_u, p_m, p_d of
:func:`optionpricing.finite_difference.price_explicit` this is exactly

    q = p / (1 - r dt),     discount factor = 1 - r dt,

so the tree and the explicit scheme produce the same numbers to rounding
error (``discount="linear"``). ``discount="exp"`` uses the more conventional
exp(-r dt); the two differ by O(dt^2) per step.
"""

from __future__ import annotations

import numpy as np

from .finite_difference import _payoff


def tree_parameters(T: float, r: float, sigma: float, n_steps: int, lam: float = 0.5):
    """Return dt, dx and the risk-neutral probabilities (q_u, q_m, q_d)."""
    dt = T / n_steps
    dx = sigma * np.sqrt(dt / lam)
    nu = r - 0.5 * sigma**2
    a = 0.5 * sigma**2 * dt / dx**2
    b = 0.5 * nu * dt / dx
    p_u, p_d = a + b, a - b
    p_m = 1.0 - 2.0 * a - r * dt
    total = p_u + p_m + p_d          # = 1 - r dt
    return dt, dx, (p_u / total, p_m / total, p_d / total)


def price_trinomial(S0: float, K: float, T: float, r: float, sigma: float,
                    kind: str = "call", american: bool = False, n_steps: int = 200,
                    lam: float = 0.5, discount: str = "linear") -> float:
    """Price by backward induction through the tree.

    American options compare continuation value with immediate exercise at
    every node.
    """
    dt, dx, (q_u, q_m, q_d) = tree_parameters(T, r, sigma, n_steps, lam)
    if min(q_u, q_m, q_d) < 0:
        raise ValueError("negative branch probability: reduce lam or increase n_steps")
    if discount == "linear":
        df = 1.0 - r * dt
    elif discount == "exp":
        df = np.exp(-r * dt)
    else:
        raise ValueError("discount must be 'linear' or 'exp'")

    j = np.arange(-n_steps, n_steps + 1)
    V = _payoff(S0 * np.exp(j * dx), K, kind)
    for n in range(n_steps - 1, -1, -1):
        V = df * (q_u * V[2:] + q_m * V[1:-1] + q_d * V[:-2])
        if american:
            S_n = S0 * np.exp(np.arange(-n, n + 1) * dx)
            V = np.maximum(V, _payoff(S_n, K, kind))
    return float(V[0])
