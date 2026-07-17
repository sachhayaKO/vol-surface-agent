"""
LangChain @tool wrappers around the pure functions in options.py and
realized_vol.py. Kept as thin wrappers on purpose: the math lives in
tools that have no framework dependency and can be tested/called
directly (see tests/), and this module's only job is turning that
output into something an LLM can read and deciding what each tool's
docstring tells the model about when to call it.
"""

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from vol_surface_agent.ingestion.build_index import EMBEDDING_MODEL, PERSIST_DIR
from vol_surface_agent.tools.options import compute_iv_for_chain, fetch_option_chain, pick_expiry_near
from vol_surface_agent.tools.realized_vol import compute_realized_vol

# Loaded lazily (only when the retriever tool is actually called), since
# instantiating the embedding model has real startup cost and the other
# two tools don't need it at all.
_retriever_store: Chroma | None = None


def _get_retriever_store() -> Chroma:
    global _retriever_store
    if _retriever_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _retriever_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="earnings_10k",
        )
    return _retriever_store


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
    except ValueError as e:
        return f"Could not fetch options data for {ticker}: {e}"

    chain_iv = compute_iv_for_chain(chain)
    spot = chain["underlying_price"].iloc[0]
    tte_days = round(chain["time_to_expiry"].iloc[0] * 365)

    near_money = chain_iv[
        (chain_iv["strike"] > spot * 0.9) & (chain_iv["strike"] < spot * 1.1)
    ].dropna(subset=["implied_vol"]).sort_values(["option_type", "strike"])

    if near_money.empty:
        return f"No solvable near-the-money implied vols for {ticker} {expiry}."

    lines = [f"{ticker} options, expiry {expiry} ({tte_days} days out), spot ${spot:.2f}:"]
    for _, row in near_money.iterrows():
        lines.append(
            f"  {row['option_type']} strike {row['strike']:.1f}: "
            f"IV {row['implied_vol']*100:.1f}%  (mid ${row['mid_price']:.2f}, vol {row['volume']:.0f})"
        )
    return "\n".join(lines)


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
    except ValueError as e:
        return f"Could not compute realized vol for {ticker}: {e}"

    lines = [f"{ticker} realized volatility (annualized):"]
    for window, estimators in sorted(result.items()):
        ctc = estimators["close_to_close"]
        park = estimators["parkinson"]
        ctc_str = f"{ctc*100:.1f}%" if ctc is not None else "n/a (insufficient history)"
        park_str = f"{park*100:.1f}%" if park is not None else "n/a (insufficient history)"
        lines.append(f"  {window}-day window: close-to-close {ctc_str}, Parkinson {park_str}")
    return "\n".join(lines)


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
    results = store.similarity_search(query, k=4, filter=filter_)

    if not results:
        scope = f" for {ticker}" if ticker else ""
        return f"No matching filing text found{scope}. Coverage is limited to AAPL and NVDA."

    lines = []
    for doc in results:
        lines.append(f"[{doc.metadata['ticker']} — {doc.metadata['source']}]")
        lines.append(doc.page_content)
        lines.append("")
    return "\n".join(lines)
