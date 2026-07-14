"""
Options chain retrieval and implied volatility calculation.

Pure functions with no framework or agent dependencies — callable directly
from a script, notebook, or (later) wrapped as a LangChain tool.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

# Flat risk-free rate used for all expiries. In reality this varies by
# maturity (a real yield curve), but for near-dated single-name options
# a flat short rate is a standard, defensible simplification. What
# actually matters is using it correctly below (forward price), not
# how precisely it's sourced.
RISK_FREE_RATE = 0.05


@dataclass
class OptionQuote:
    """One row of an options chain, trimmed to what the vol math needs."""

    ticker: str
    expiry: str
    strike: float
    option_type: str  # "call" or "put"
    mid_price: float
    underlying_price: float
    time_to_expiry: float  # in years, ACT/365
    dividend_yield: float


def fetch_option_chain(ticker: str, expiry: str | None = None) -> pd.DataFrame:
    """
    Fetch the options chain for a single expiry.

    If expiry is None, uses the nearest available expiry. Returns a
    DataFrame with both calls and puts, plus underlying spot price and
    time to expiry attached to every row so downstream IV calcs are
    self-contained.

    Raises ValueError if the ticker has no listed options or the
    requested expiry doesn't exist.
    """
    tk = yf.Ticker(ticker)

    expiries = tk.options
    if not expiries:
        raise ValueError(f"{ticker} has no listed options (or ticker is invalid)")

    if expiry is None:
        expiry = expiries[0]
    elif expiry not in expiries:
        raise ValueError(
            f"{expiry} is not a valid expiry for {ticker}. "
            f"Available: {expiries[:5]}{'...' if len(expiries) > 5 else ''}"
        )

    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    calls["option_type"] = "call"
    puts = chain.puts.copy()
    puts["option_type"] = "put"
    df = pd.concat([calls, puts], ignore_index=True)

    if df.empty:
        raise ValueError(f"{ticker} {expiry} returned an empty option chain")

    spot = tk.fast_info["last_price"]
    if spot is None or spot <= 0:
        raise ValueError(f"could not get a valid spot price for {ticker}")

    # Dividend yield, derived from dollar dividend / spot rather than
    # trusting tk.info["dividendYield"] directly: that field's units have
    # changed across yfinance versions (sometimes a decimal fraction like
    # 0.006, sometimes a percent-point number like 0.6 meaning "0.6%"),
    # and silently picking the wrong one produces a several-point error
    # in the forward price. dividendRate (a dollar amount) is stable.
    dividend_rate = tk.info.get("dividendRate") or 0.0
    dividend_yield = dividend_rate / spot if dividend_rate else 0.0

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    time_to_expiry = max((expiry_date - now).days, 1) / 365.0

    df["underlying_price"] = spot
    df["time_to_expiry"] = time_to_expiry
    df["dividend_yield"] = dividend_yield
    df["expiry"] = expiry
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    # some strikes have no live quotes (bid/ask both 0); fall back to
    # lastPrice rather than silently keeping a mid of 0.
    zero_mid = df["mid_price"] <= 0
    df.loc[zero_mid, "mid_price"] = df.loc[zero_mid, "lastPrice"]

    return df[
        [
            "expiry",
            "strike",
            "option_type",
            "bid",
            "ask",
            "mid_price",
            "lastPrice",
            "volume",
            "openInterest",
            "underlying_price",
            "time_to_expiry",
            "dividend_yield",
        ]
    ]


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    option_type: str,
) -> float:
    """
    European Black-Scholes-Merton price.

    Note: yfinance option chains are American-style equity options
    (early exercise allowed), but this model assumes European exercise.
    The gap is usually small for non-dividend payers and short-dated
    options; it grows for dividend payers around ex-div dates and for
    deep ITM puts. This is a known, accepted approximation for this
    project, not an oversight.
    """
    if time_to_expiry <= 0 or vol <= 0:
        intrinsic = (
            max(spot - strike, 0.0)
            if option_type == "call"
            else max(strike - spot, 0.0)
        )
        return intrinsic

    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * vol**2) * time_to_expiry
    ) / (vol * np.sqrt(time_to_expiry))
    d2 = d1 - vol * np.sqrt(time_to_expiry)

    disc_q = np.exp(-dividend_yield * time_to_expiry)
    disc_r = np.exp(-rate * time_to_expiry)

    if option_type == "call":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    else:
        return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
) -> float | None:
    """
    Solve for implied vol via Brent's method (bracketed root-finding on
    price(vol) - market_price = 0), not a crude bisection or fixed-step
    search. Brent's method combines bisection's guaranteed convergence
    with faster interpolation steps when the function is well-behaved.

    Returns None if no solution exists in [1e-4, 5.0] (500% vol) — this
    happens for prices below intrinsic value or above the max possible
    value, usually from stale/bad quotes rather than a real market IV.
    """

    def price_diff(vol: float) -> float:
        return (
            black_scholes_price(
                spot, strike, time_to_expiry, rate, dividend_yield, vol, option_type
            )
            - market_price
        )

    try:
        return brentq(price_diff, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        # price_diff doesn't change sign across the bracket => the
        # market price is outside what any vol in [0.01%, 500%] can
        # produce for this contract. Treat as "not solvable" rather
        # than raising, since bad single quotes are routine in a chain.
        return None


def compute_iv_for_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Attach an implied_vol column to a chain DataFrame (as returned by
    fetch_option_chain). Rows where IV can't be solved get NaN, not a
    dropped row, so the caller can see how much of the chain failed.
    """
    df = chain.copy()

    def solve_row(row: pd.Series) -> float | None:
        return implied_vol(
            market_price=row["mid_price"],
            spot=row["underlying_price"],
            strike=row["strike"],
            time_to_expiry=row["time_to_expiry"],
            rate=RISK_FREE_RATE,
            dividend_yield=row["dividend_yield"],
            option_type=row["option_type"],
        )

    df["implied_vol"] = df.apply(solve_row, axis=1)
    return df


def pick_expiry_near(ticker: str, target_days: int = 35) -> str:
    """
    Pick the listed expiry closest to target_days out. The *nearest*
    expiry (0-3 days out) is often too close to expiration: deep ITM/OTM
    strikes go untraded, bid/ask quotes go stale, and the IV solver
    produces noisy garbage on those rows. A ~30-45 day expiry is a more
    representative surface for eyeballing sanity.
    """
    tk = yf.Ticker(ticker)
    today = datetime.now(timezone.utc)
    return min(
        tk.options,
        key=lambda e: abs(
            (datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) - today).days
            - target_days
        ),
    )
