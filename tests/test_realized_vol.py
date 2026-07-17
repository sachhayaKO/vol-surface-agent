"""
Unit tests for realized vol estimators. No network calls — synthetic
price series only.
"""

import numpy as np
import pandas as pd
import pytest

from vol_surface_agent.tools.realized_vol import (
    TRADING_DAYS_PER_YEAR,
    close_to_close_vol,
    parkinson_vol,
)


def test_close_to_close_vol_recovers_known_stdev():
    """Generate returns with a known daily stdev and check the annualized
    output matches daily_stdev * sqrt(252)."""
    rng = np.random.default_rng(seed=42)
    daily_vol = 0.02
    log_returns = rng.normal(loc=0, scale=daily_vol, size=500)
    prices = pd.Series(100 * np.exp(np.cumsum(log_returns)))

    result = close_to_close_vol(prices, window=250)

    assert result == pytest.approx(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR), rel=0.1)


def test_close_to_close_vol_none_when_insufficient_history():
    prices = pd.Series([100.0, 101.0, 99.0])
    assert close_to_close_vol(prices, window=30) is None


def test_parkinson_vol_zero_when_high_equals_low():
    """If high == low every day (no intraday range at all), Parkinson
    vol should be exactly zero — there's no measured intraday movement."""
    high = pd.Series([100.0] * 30)
    low = pd.Series([100.0] * 30)
    assert parkinson_vol(high, low, window=20) == 0.0


def test_parkinson_vol_none_when_insufficient_history():
    high = pd.Series([100.0, 101.0])
    low = pd.Series([99.0, 100.0])
    assert parkinson_vol(high, low, window=20) is None
