"""Closed-form Black-Scholes prices and Greeks for European options.

All functions accept NumPy arrays and broadcast, so a whole strike or spot grid
can be priced in one call. ``kind`` is ``"call"`` or ``"put"``. There are no
dividends: the underlying follows dS = r S dt + sigma S dW under the
risk-neutral measure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def d1_d2(S, K, T, r, sigma):
    """Return (d1, d2) from the Black-Scholes formula."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, kind: str = "call"):
    """Black-Scholes price of a European call or put."""
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    disc_K = K * np.exp(-r * np.asarray(T))
    if kind == "call":
        return S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    return disc_K * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(S, K, T, r, sigma, kind: str = "call"):
    """dV/dS."""
    _check_kind(kind)
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma):
    """d2V/dS2 (identical for calls and puts)."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """dV/dsigma (identical for calls and puts), per unit of volatility."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, kind: str = "call"):
    """dV/dt (calendar time), per year. Usually negative for a long option."""
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    disc_K = K * np.exp(-r * np.asarray(T))
    if kind == "call":
        return decay - r * disc_K * norm.cdf(d2)
    return decay + r * disc_K * norm.cdf(-d2)


def rho(S, K, T, r, sigma, kind: str = "call"):
    """dV/dr."""
    _check_kind(kind)
    _, d2 = d1_d2(S, K, T, r, sigma)
    disc_K = K * np.exp(-r * np.asarray(T))
    if kind == "call":
        return T * disc_K * norm.cdf(d2)
    return -T * disc_K * norm.cdf(-d2)


def implied_vol(target_price: float, S: float, K: float, T: float, r: float,
                kind: str = "call", lo: float = 1e-6, hi: float = 5.0) -> float:
    """Volatility that reproduces ``target_price`` (Brent's method)."""
    return brentq(lambda s: price(S, K, T, r, s, kind) - target_price, lo, hi, xtol=1e-12)
