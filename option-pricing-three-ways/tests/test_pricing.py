"""Correctness checks. Run with ``pytest``."""

import numpy as np
import pytest

from optionpricing import black_scholes as bs
from optionpricing import finite_difference as fd
from optionpricing import trinomial as tri

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


def test_known_value():
    # Hull, Options Futures and Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
    assert bs.price(S0, K, T, R, SIG) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("K_", [80.0, 100.0, 125.0])
def test_put_call_parity(K_):
    lhs = bs.price(S0, K_, T, R, SIG, "call") - bs.price(S0, K_, T, R, SIG, "put")
    assert lhs == pytest.approx(S0 - K_ * np.exp(-R * T), abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_bumped_prices(kind):
    h = 1e-4
    p = lambda S=S0, s=SIG, r=R: bs.price(S, K, T, r, s, kind)
    assert bs.delta(S0, K, T, R, SIG, kind) == pytest.approx((p(S0 + h) - p(S0 - h)) / (2 * h), rel=1e-6)
    assert bs.gamma(S0, K, T, R, SIG) == pytest.approx((p(S0 + h) - 2 * p() + p(S0 - h)) / h**2, rel=1e-3)
    assert bs.vega(S0, K, T, R, SIG) == pytest.approx((p(s=SIG + h) - p(s=SIG - h)) / (2 * h), rel=1e-6)
    assert bs.rho(S0, K, T, R, SIG, kind) == pytest.approx((p(r=R + h) - p(r=R - h)) / (2 * h), rel=1e-6)
    # theta is the derivative in calendar time, i.e. minus the derivative in maturity
    dT = (bs.price(S0, K, T + h, R, SIG, kind) - bs.price(S0, K, T - h, R, SIG, kind)) / (2 * h)
    assert bs.theta(S0, K, T, R, SIG, kind) == pytest.approx(-dT, rel=1e-6)


def test_black_scholes_pde_holds():
    """theta + 0.5 sigma^2 S^2 gamma + r S delta - r V = 0."""
    S = np.linspace(60, 160, 11)
    lhs = (bs.theta(S, K, T, R, SIG) + 0.5 * SIG**2 * S**2 * bs.gamma(S, K, T, R, SIG)
           + R * S * bs.delta(S, K, T, R, SIG) - R * bs.price(S, K, T, R, SIG))
    assert np.abs(lhs).max() < 1e-10


def test_implied_vol_round_trip():
    target = bs.price(S0, 110, 0.5, R, 0.31, "put")
    assert bs.implied_vol(target, S0, 110, 0.5, R, "put") == pytest.approx(0.31, abs=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_explicit_converges(kind):
    exact = bs.price(S0, K, T, R, SIG, kind)
    assert fd.price_explicit(S0, K, T, R, SIG, kind, n_steps=2000).price == pytest.approx(exact, abs=1e-3)


def test_crank_nicolson_converges_second_order():
    exact = bs.price(S0, K, T, R, SIG)
    e1 = abs(fd.price_theta(S0, K, T, R, SIG, n_steps=100, n_space=400, rannacher_steps=2).price - exact)
    e2 = abs(fd.price_theta(S0, K, T, R, SIG, n_steps=200, n_space=800, rannacher_steps=2).price - exact)
    assert e2 < 1e-3
    assert 3.0 < e1 / e2 < 5.0  # halving the grid cuts the error by about four


@pytest.mark.parametrize("kind,american", [("call", False), ("put", False), ("put", True)])
def test_tree_equals_explicit_scheme(kind, american):
    N = 250
    v_fd = fd.price_explicit(S0, K, T, R, SIG, kind, american, n_steps=N, n_half=N).price
    v_tr = tri.price_trinomial(S0, K, T, R, SIG, kind, american, n_steps=N)
    assert v_fd == pytest.approx(v_tr, abs=1e-11)


def test_tree_probabilities_match_moments():
    N = 400
    dt, dx, (qu, qm, qd) = tri.tree_parameters(T, R, SIG, N)
    assert qu + qm + qd == pytest.approx(1.0)
    # both moments hold up to a factor 1/(1 - r dt), i.e. to leading order in dt
    assert (qu - qd) * dx == pytest.approx((R - 0.5 * SIG**2) * dt / (1 - R * dt), rel=1e-12)
    assert (qu + qd) * dx**2 == pytest.approx(SIG**2 * dt / (1 - R * dt), rel=1e-12)


def test_american_put_known_value_and_bounds():
    am = tri.price_trinomial(S0, K, T, R, SIG, "put", True, n_steps=2000, discount="exp")
    assert am == pytest.approx(6.0904, abs=2e-3)   # published benchmark value
    assert am > bs.price(S0, K, T, R, SIG, "put")
    deep = tri.price_trinomial(S0, 150.0, T, R, SIG, "put", True, n_steps=500)
    assert deep >= 150.0 - S0 - 1e-12             # never worth less than exercising now


def test_american_call_without_dividends_is_european():
    am = tri.price_trinomial(S0, K, T, R, SIG, "call", True, n_steps=1000, discount="exp")
    eu = tri.price_trinomial(S0, K, T, R, SIG, "call", False, n_steps=1000, discount="exp")
    assert am == pytest.approx(eu, abs=1e-12)


def test_grid_greeks():
    res = fd.price_explicit(S0, K, T, R, SIG, n_steps=3200)
    assert res.delta == pytest.approx(bs.delta(S0, K, T, R, SIG), abs=1e-4)
    assert res.gamma == pytest.approx(bs.gamma(S0, K, T, R, SIG), abs=1e-5)


def test_explicit_scheme_unstable_when_middle_weight_negative():
    res = fd.price_explicit(S0, K, T, R, SIG, n_steps=400, lam=1.05)
    assert res.p_mid < 0
    assert abs(res.price - bs.price(S0, K, T, R, SIG)) > 1e3


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        bs.price(S0, K, T, R, SIG, "straddle")
