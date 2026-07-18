"""
Shared text formatting for tool output. Pulled out of langchain_tools.py
so both the Milestone 2 tools (catch errors, return a friendly string)
and the Milestone 3 tools (raise typed errors, let the graph decide what
to do) can produce identical output on the success path — the only
difference between the two is how failure is handled, not how a good
result gets formatted.
"""

import pandas as pd


def format_iv_surface(chain_iv: pd.DataFrame, ticker: str, expiry: str, spot: float, tte_days: int) -> str | None:
    """Format near-the-money implied vols as a readable table. Returns
    None if there's nothing solvable to show, so the caller decides how
    to treat that (friendly message vs. raised error)."""
    near_money = chain_iv[
        (chain_iv["strike"] > spot * 0.9) & (chain_iv["strike"] < spot * 1.1)
    ].dropna(subset=["implied_vol"]).sort_values(["option_type", "strike"])

    if near_money.empty:
        return None

    lines = [f"{ticker} options, expiry {expiry} ({tte_days} days out), spot ${spot:.2f}:"]
    for _, row in near_money.iterrows():
        lines.append(
            f"  {row['option_type']} strike {row['strike']:.1f}: "
            f"IV {row['implied_vol']*100:.1f}%  (mid ${row['mid_price']:.2f}, vol {row['volume']:.0f})"
        )
    return "\n".join(lines)


def format_realized_vol(ticker: str, result: dict) -> str:
    """Format realized vol estimators across windows as a readable table."""
    lines = [f"{ticker} realized volatility (annualized):"]
    for window, estimators in sorted(result.items()):
        ctc = estimators["close_to_close"]
        park = estimators["parkinson"]
        ctc_str = f"{ctc*100:.1f}%" if ctc is not None else "n/a (insufficient history)"
        park_str = f"{park*100:.1f}%" if park is not None else "n/a (insufficient history)"
        lines.append(f"  {window}-day window: close-to-close {ctc_str}, Parkinson {park_str}")
    return "\n".join(lines)


def format_filing_results(results: list, ticker: str | None) -> str | None:
    """Format retrieved filing chunks. Returns None if there's nothing
    to show, so the caller decides how to treat that."""
    if not results:
        return None

    lines = []
    for doc in results:
        lines.append(f"[{doc.metadata['ticker']} — {doc.metadata['source']}]")
        lines.append(doc.page_content)
        lines.append("")
    return "\n".join(lines)
