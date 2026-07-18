"""
Milestone 3 tool variants: same tools as tools/langchain_tools.py, same
docstrings (the model's view of when/how to call them doesn't change),
but failures raise typed errors (see agent/errors.py) instead of being
caught and turned into a friendly string.

This is the deliberate difference from Milestone 2: create_agent has no
way to branch graph execution on *why* a tool failed, so Milestone 2's
tools swallow errors and let the model react to error text in prose.
The custom StateGraph in agent/graph.py *can* branch explicitly, so
these tools let it — errors propagate up to execute_tool, which
classifies and routes on them as real state, not prose the model has to
interpret consistently on its own.
"""

from typing import NoReturn

from langchain_core.tools import tool

from vol_surface_agent.agent.errors import ApiFailureError, BadTickerError, EmptyDataError
from vol_surface_agent.tools.formatting import format_filing_results, format_iv_surface, format_realized_vol
from vol_surface_agent.tools.langchain_tools import _get_retriever_store
from vol_surface_agent.tools.options import compute_iv_for_chain, fetch_option_chain, pick_expiry_near
from vol_surface_agent.tools.realized_vol import compute_realized_vol


def _raise_typed(e: Exception) -> NoReturn:
    """
    Classify a caught exception into one of the typed errors in
    agent/errors.py and raise it.

    A ValueError from fetch_option_chain/fetch_price_history almost
    always means a bad/unsupported ticker (see their error messages in
    tools/options.py and tools/realized_vol.py) — except the specific
    "empty option chain" case, which means the ticker is fine but this
    particular expiry/query returned nothing. Anything else (network
    errors, unexpected failures) is treated as a transient API failure,
    since "retry" is the safest default reaction to an error this code
    didn't specifically anticipate.
    """
    if isinstance(e, ValueError):
        if "empty option chain" in str(e):
            raise EmptyDataError(str(e)) from e
        raise BadTickerError(str(e)) from e
    raise ApiFailureError(str(e)) from e


@tool
def get_implied_vol_surface(ticker: str) -> str:
    """
    Fetch a live options chain for the given stock ticker and compute
    implied volatility near the money, for an expiry roughly a month out.

    Use this when the user asks about implied volatility, options
    pricing, or the volatility "surface" or "skew" for a specific
    ticker. Returns near-the-money call and put implied vols by strike,
    which shows the skew directly (puts vs calls at the same strike,
    across strikes).

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    try:
        expiry = pick_expiry_near(ticker)
        chain = fetch_option_chain(ticker, expiry=expiry)
    except Exception as e:
        _raise_typed(e)

    chain_iv = compute_iv_for_chain(chain)
    spot = chain["underlying_price"].iloc[0]
    tte_days = round(chain["time_to_expiry"].iloc[0] * 365)

    formatted = format_iv_surface(chain_iv, ticker, expiry, spot, tte_days)
    if formatted is None:
        raise EmptyDataError(f"No solvable near-the-money implied vols for {ticker} {expiry}.")
    return formatted


@tool
def get_realized_vol(ticker: str) -> str:
    """
    Compute historical (realized) volatility for a stock ticker over
    several lookback windows (10, 20, and 60 trading days), using two
    different estimators: close-to-close (daily closing prices only) and
    Parkinson (uses the daily high/low range, captures more intraday
    movement).

    Use this when the user asks how volatile a stock has actually been
    recently, or wants to compare implied vol against realized vol to
    judge whether options look "rich" or "cheap".

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    try:
        result = compute_realized_vol(ticker)
    except Exception as e:
        _raise_typed(e)

    return format_realized_vol(ticker, result)


@tool
def search_earnings_and_filings(query: str, ticker: str | None = None) -> str:
    """
    Search recent earnings press releases and 10-K risk factor sections
    for grounding context. Use this when the user asks *why* something
    might be happening (e.g. "is there a reason vol looks elevated right
    now", "what did the company say about risks/guidance"), not for
    numeric vol/options questions.

    Coverage is currently limited to AAPL and NVDA — SPY is an ETF with
    no 10-K or earnings filings of its own, so it has no results here.

    Args:
        query: Natural-language search query, e.g. "supply chain risk"
            or "guidance for next quarter".
        ticker: Optional ticker to restrict results to (e.g. "AAPL").
            Leave unset to search across all covered tickers.
    """
    store = _get_retriever_store()
    filter_ = {"ticker": ticker.upper()} if ticker else None
    try:
        results = store.similarity_search(query, k=4, filter=filter_)
    except Exception as e:
        raise ApiFailureError(str(e)) from e

    formatted = format_filing_results(results, ticker)
    if formatted is None:
        scope = f" for {ticker}" if ticker else ""
        raise EmptyDataError(f"No matching filing text found{scope}.")
    return formatted


GRAPH_TOOLS = [get_implied_vol_surface, get_realized_vol, search_earnings_and_filings]
