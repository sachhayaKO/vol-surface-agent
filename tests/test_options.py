"""
Unit tests for the options vol math. No network calls — these test the
math in isolation against synthetic prices, so they run fast and
deterministically in CI.
"""

import numpy as np
import pytest

from vol_surface_agent.tools.options import black_scholes_price, implied_vol


def test_implied_vol_round_trips_black_scholes_price():
    """Pricing at a known vol, then solving IV from that price, should
    recover the original vol. This is the core correctness check for the
    Brent's-method solver."""
    true_vol = 0.28
    price = black_scholes_price(
        spot=100.0,
        strike=105.0,
        time_to_expiry=0.25,
        rate=0.05,
        dividend_yield=0.01,
        vol=true_vol,
        option_type="call",
    )

    solved_vol = implied_vol(
        market_price=price,
        spot=100.0,
        strike=105.0,
        time_to_expiry=0.25,
        rate=0.05,
        dividend_yield=0.01,
        option_type="call",
    )

    assert solved_vol == pytest.approx(true_vol, abs=1e-4)


def test_implied_vol_returns_none_for_unsolvable_price():
    """A price below intrinsic value has no vol that can produce it —
    the solver should report that explicitly (None) rather than raise
    or silently return a garbage number."""
    solved_vol = implied_vol(
        market_price=-5.0,  # not a real price; guaranteed outside the bracket
        spot=100.0,
        strike=105.0,
        time_to_expiry=0.25,
        rate=0.05,
        dividend_yield=0.0,
        option_type="call",
    )
    assert solved_vol is None


def test_put_call_parity_holds_at_a_given_vol():
    """Regression test for a real bug: a dividend-yield unit mismatch
    (see docs/DEVLOG.md, 2026-07-13) broke put-call parity by ~$10 on a
    $300 stock without raising any error — the IV solver just silently
    returned wrong numbers for puts vs calls at the same strike. Parity
    is a cheap, always-true identity, so checking it here catches the
    same class of bug (bad rate/dividend/forward inputs) even if the
    root cause next time is different from last time.

    C - P = S*e^(-qT) - K*e^(-rT)
    """
    spot, strike, tte, rate, q = 300.0, 310.0, 37 / 365, 0.05, 0.0034

    call_price = black_scholes_price(spot, strike, tte, rate, q, vol=0.30, option_type="call")
    put_price = black_scholes_price(spot, strike, tte, rate, q, vol=0.30, option_type="put")

    lhs = call_price - put_price
    rhs = spot * np.exp(-q * tte) - strike * np.exp(-rate * tte)

    assert lhs == pytest.approx(rhs, abs=1e-6)
