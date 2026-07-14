# Learning Log

Running notes on what's being built and why, in plain language, so this
project is understandable and not just AI output. Newest entries at the
bottom.

---

## 2026-07-13 — Project kickoff

**Decisions locked in:**
- Phase order is fixed: (1) raw hand-written tool, (2) LangChain +
  `create_agent`, (3) custom LangGraph `StateGraph`, (4) evals + LangSmith.
  No skipping ahead.
- Tickers for phase 1/2 dev: a fixed small list of liquid single names
  (AAPL, SPY, NVDA) rather than arbitrary user input — keeps option-chain
  data quirks predictable while the vol math is being built.
- Risk-free rate / dividend yield: flat constants (not a full treasury
  curve). This is standard practice for near-dated single-name options.
  The important part isn't the curve, it's *using the rate correctly* —
  implied vol should be solved off the forward price
  `F = S * e^((r-q)*T)`, not naive spot.
- Where "options knowledge" effort actually goes instead of a rate curve:
  - A real IV root-finder (Brent's method), not a crude bisection.
  - Acknowledging the American vs. European exercise mismatch (yfinance
    chains are American-style; Black-Scholes assumes European).
  - Offering more than one realized-vol estimator later (close-to-close
    and Parkinson), since the choice of estimator is itself a real
    richness/cheapness design decision.
  - Reasoning about skew in moneyness/delta space, not raw strike.
- Phase 2 earnings/10-K text source: scraped/downloaded automatically
  (e.g. SEC EDGAR), not manually pasted. Adds more scope to phase 2 than
  the minimal option, by design choice.
- LLM provider: Anthropic Claude via `langchain_anthropic`.

**Project structure created** (directories only so far):
```
phase1_raw_tools/       hand-written functions, no framework
phase2_langchain_agent/ @tool-wrapped tools + create_agent
phase3_langgraph/       custom StateGraph
phase4_evals/           golden dataset, scoring scripts, LangSmith
data/                   local Chroma store, cached earnings/10-K text
tests/                  unit tests for the vol math, shared across phases
```

**Environment:** system Python is 3.14.5 via Homebrew, no packages
installed yet. Setting up a project-local virtualenv (`.venv`) rather than
installing into system Python, so dependencies stay isolated per project.

**Next:** Phase 1 — a hand-written Python function (no framework) that
fetches an options chain via `yfinance` and computes implied vol for a
single expiry.

---

## 2026-07-13 — Phase 1: raw IV tool

**Environment:** created a project-local virtualenv (`.venv/`) rather than
installing into system Python. Installed `yfinance`, `numpy`, `scipy`,
`pandas`.

**Built `phase1_raw_tools/vol_tools.py`**, no LangChain/agent framework:
- `fetch_option_chain(ticker, expiry)` — pulls calls+puts for one expiry
  from yfinance, attaches spot price, time-to-expiry, dividend yield to
  every row. Falls back to `lastPrice` when bid/ask are both 0 (illiquid
  strikes) instead of computing a mid of 0.
- `black_scholes_price(...)` — European BSM price. Documented the known
  approximation: yfinance chains are American-style, BSM assumes
  European; the gap matters most for ITM puts near ex-dividend dates.
- `implied_vol(...)` — solves for IV via **Brent's method**
  (`scipy.optimize.brentq`) bracketed on `[0.01%, 500%]` vol, not a
  crude bisection. Returns `None` (not an exception) when no solution
  exists in that bracket, since a handful of bad quotes in a chain is
  normal and shouldn't kill the whole calc.
- `compute_iv_for_chain(...)` — applies IV solving across a full chain,
  keeping unsolved rows as NaN so you can see the solve rate.

**Bug found and fixed while sanity-checking the first real run:**
The first run (AAPL, nearest expiry) showed wildly inconsistent IVs —
some contracts at 170%+ vol. Two separate issues, found by actually
inspecting the output instead of assuming it worked:

1. **Expiry choice.** The "nearest" expiry was only ~1 day out. At that
   horizon, illiquid deep ITM/OTM strikes have stale bid/ask quotes and
   the IV solver produces noise on them. Fixed by adding
   `_pick_expiry_near()`, which picks the listed expiry closest to a
   target (default 35 days) instead of always using the nearest one —
   more representative of a "normal" surface.

2. **Dividend yield unit bug (the real one).** Even on a cleaner 37-day
   expiry, call and put IV at the *same strike* disagreed by 30+ vol
   points (e.g. strike 320: call IV 40% vs put IV 8%) — checked this by
   testing put-call parity (`C - P` should ≈ `S*e^(-qT) - K*e^(-rT)`)
   directly against raw market mid prices, independent of the IV solver.
   The parity gap was a *constant* ~$10 offset across every strike,
   which pointed at a broken input rather than random bad data.
   Root cause: `yfinance`'s `tk.info["dividendYield"]` field returned
   `0.34` for AAPL — but that's `0.34%` in this yfinance version, not a
   0.34 decimal fraction (i.e. 34%). My original code only rescaled the
   field when it was `> 1`, which missed this case entirely. Fixed by
   computing dividend yield from `dividendRate / spot` instead (a dollar
   amount divided by price), which doesn't depend on yfinance's yield
   field formatting at all. After the fix, put-call parity holds to
   within a few cents, and call/put IV at the same strike now agree
   within ~1 point (the small remaining gap is the expected American
   early-exercise premium on puts, not an error).

   **Takeaway:** don't trust a single vendor field's units — when
   something is derivable two ways (yield field vs. dollar rate ÷
   price), cross-check them, and use put-call parity as a free
   sanity check on the whole pipeline (spot, rate, dividend, and
   quotes) before trusting any individual IV number.

**Result:** `python phase1_raw_tools/vol_tools.py` runs end to end
against live AAPL data — fetches a ~35-day chain, solves IV for
121/134 contracts, and prints a near-the-money slice showing a
coherent downside skew (~35% deep OTM puts down to ~27% near ATM/calls).

**Next:** hold here for review before starting Phase 2 (wrapping this as
LangChain `@tool`s + `create_agent`).

---

## 2026-07-13 — Restructure: from phase folders to a real package layout

The `phase1_raw_tools/`, `phase2_langchain_agent/` etc. directory names
were fine as a personal build order but read as tutorial scaffolding, not
a project — not something you'd want a reviewer's first impression of the
repo to be. Restructured into a standard `src/` package layout so the
repo looks and installs like a normal Python project; the milestone-based
build order still exists, just moved into documentation
([ARCHITECTURE.md](ARCHITECTURE.md)) instead of the folder names.

Changes:
- `phase1_raw_tools/vol_tools.py` → `src/vol_surface_agent/tools/options.py`
  (code unchanged, just relocated + module docstring de-"phase"-ified).
- Added `pyproject.toml` (setuptools, `src/` layout) so the package
  installs with `pip install -e ".[dev]"` instead of running scripts
  against a loose folder. Dropped `requirements.txt` in favor of this.
- Added `scripts/demo_iv.py` — the old `if __name__ == "__main__"` block,
  now a proper standalone script that imports the installed package.
- Added `tests/test_options.py` with real unit tests, including turning
  the dividend-yield/put-call-parity bug (above) into a permanent
  regression test — parity is a cheap, always-true identity, so checking
  it in CI catches the same *class* of bug (bad rate/dividend/forward
  inputs) even when the next root cause is different.
- Added empty `src/vol_surface_agent/agent/` and `.../ingestion/`
  packages as placeholders for Milestones 2-3, and `evals/` for
  Milestone 4, so the package structure doesn't need another reshuffle
  later.
- This dev log moved from `LEARNING_LOG.md` at the repo root to
  `docs/DEVLOG.md`.

Verified after the move: `pytest` (3/3 pass) and
`python scripts/demo_iv.py AAPL` both work identically to before against
the new installed-package layout.

**Next:** Milestone 2 — wrap `tools/options.py` as LangChain `@tool`s,
add a realized-vol tool and a Chroma retriever tool, wire into
`create_agent`.
