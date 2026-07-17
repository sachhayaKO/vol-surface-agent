"""
Realized volatility estimators. Pure functions, no framework dependency —
same pattern as tools/options.py.
"""

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def fetch_price_history(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch daily OHLC price history for a ticker. Pulls extra calendar days
    beyond lookback_days to account for weekends/holidays, then returns
    whatever trading days actually came back (does not pad or interpolate
    missing days).

    Raises ValueError if yfinance returns no data (bad ticker, delisted,
    etc).
    """
    calendar_days = int(lookback_days * 1.6) + 10
    hist = yf.Ticker(ticker).history(period=f"{calendar_days}d", interval="1d")
    if hist.empty:
        raise ValueError(f"no price history returned for {ticker}")
    return hist[["Open", "High", "Low", "Close"]]


def close_to_close_vol(prices: pd.Series, window: int) -> float | None:
    """
    Naive close-to-close realized vol: stdev of daily log returns over the
    trailing `window` trading days, annualized. Returns None if there
    isn't enough history for the requested window.
    """
    if len(prices) < window + 1:
        return None
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return float(log_returns.tail(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> float | None:
    """
    Parkinson realized vol: uses the daily high/low range instead of just
    the close, so it captures intraday movement close-to-close throws
    away. More efficient than close-to-close when there's no overnight
    gap/drift; less reliable across earnings-type gap moves, which is
    exactly why this project offers both rather than picking one and
    treating it as ground truth — see the "realized vol" concept note.

    Formula: sqrt( (1 / (4*ln(2)*window)) * sum( ln(H_t/L_t)^2 ) ), annualized.
    """
    if len(high) < window or len(low) < window:
        return None
    log_hl = np.log(high / low).tail(window)
    variance = (log_hl**2).sum() / (4 * np.log(2) * window)
    return float(np.sqrt(variance) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_realized_vol(ticker: str, windows: list[int] | None = None) -> dict:
    """
    Compute both estimators across a set of lookback windows for a
    ticker. Returns a dict keyed by window, each holding both estimator
    values, so the caller can see how sensitive the read is to both the
    window and the estimator choice rather than trusting a single number.
    """
    if windows is None:
        windows = [10, 20, 60]

    max_window = max(windows)
    hist = fetch_price_history(ticker, lookback_days=max_window + 20)

    result = {}
    for w in windows:
        result[w] = {
            "close_to_close": close_to_close_vol(hist["Close"], w),
            "parkinson": parkinson_vol(hist["High"], hist["Low"], w),
        }
    return result
