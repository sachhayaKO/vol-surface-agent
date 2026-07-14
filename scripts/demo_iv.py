"""
Manual run/demo: fetch a live options chain and print implied vol for the
near-the-money contracts. Not a test — a script to eyeball real output.

Usage:
    python scripts/demo_iv.py [TICKER]
"""

import sys

from vol_surface_agent.tools.options import (
    compute_iv_for_chain,
    fetch_option_chain,
    pick_expiry_near,
)


def main(ticker: str = "AAPL") -> None:
    expiry = pick_expiry_near(ticker)
    print(f"Fetching option chain for {ticker}, expiry={expiry}...")
    chain = fetch_option_chain(ticker, expiry=expiry)
    spot = chain["underlying_price"].iloc[0]
    tte = chain["time_to_expiry"].iloc[0]
    print(f"  expiry={expiry}  spot={spot:.2f}  time_to_expiry={tte:.4f}y")
    print(f"  {len(chain)} contracts fetched")

    chain_iv = compute_iv_for_chain(chain)

    solved = chain_iv["implied_vol"].notna().sum()
    print(f"  IV solved for {solved}/{len(chain_iv)} contracts")

    near_money = chain_iv[
        (chain_iv["strike"] > spot * 0.85) & (chain_iv["strike"] < spot * 1.15)
    ].sort_values(["option_type", "strike"])

    print("\nNear-the-money contracts:")
    print(
        near_money[
            ["option_type", "strike", "mid_price", "implied_vol", "volume"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
