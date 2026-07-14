# Architecture

## What this is

An agent for options volatility analysis: it pulls a live options chain,
computes implied vol and greeks, retrieves grounding context from recent
earnings calls / 10-Ks, and answers questions about skew and whether vol
looks rich or cheap versus realized vol.

## System overview

```
                 ┌─────────────────────┐
                 │   LangGraph agent    │
                 │  (reasoning + tool   │
                 │   routing, src/      │
                 │   vol_surface_agent/ │
                 │   agent/)            │
                 └─────────┬────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────────┐
│ options chain  │  │  realized vol   │  │  earnings/10-K    │
│ + implied vol  │  │  calculator     │  │  retriever         │
│ (tools/         │  │  (tools/)       │  │  (ingestion/ +      │
│  options.py)   │  │                 │  │   Chroma store)     │
└───────────────┘  └────────────────┘  └──────────────────┘
        │
        ▼
   yfinance (live market data)
```

## Build plan

This was built in four deliberate milestones, each reviewed before moving
to the next. The milestone history is preserved here (and in detail in
[DEVLOG.md](DEVLOG.md)) because *how* it was built is part of the point of
this project — going straight to a framework would have hidden a lot of
the actual engineering decisions below.

### Milestone 1 — Vol math, no framework
Hand-written functions (`src/vol_surface_agent/tools/options.py`): fetch
an options chain, price with Black-Scholes, solve implied vol via Brent's
method. No agent loop — the goal was to understand exactly what the tool
does before any framework wraps it, and to shake out real data-quality
bugs (see DEVLOG) before they're hidden behind an LLM's tool-calling.
**Status: done.**

### Milestone 2 — LangChain tools + `create_agent`
Wrap Milestone 1 logic as LangChain tools (`@tool` decorator, clear
docstrings for tool selection), add a realized-vol tool and a retriever
tool over a small Chroma store of earnings call / 10-K text ingested from
SEC EDGAR. Wire into `create_agent` with an `InMemorySaver` checkpointer
and a `thread_id` for multi-turn state. **Status: not started.**

### Milestone 3 — Custom LangGraph state machine
Rebuild the agent as an explicit `StateGraph`: a reasoning node, a
tool-execution node, conditional routing on whether more data is needed.
Explicit state transitions (not silent failures) for: a failed API call,
a bad/missing ticker, a tool returning empty data, and malformed tool
args from the model. Same checkpointer, so state survives a crash
mid-run. **Status: not started.**

### Milestone 4 — Evals and observability
A golden dataset of 30-50 questions with known-correct or known-range
answers, scored with both LLM-as-judge and deterministic checks (right
tool called, number in the right ballpark). LangSmith tracing across
every agent turn. A documented, deliberately-induced failure mode
diagnosed via a LangSmith trace. **Status: not started.**

## Key design decisions

- **Flat risk-free rate**, not a full treasury curve — standard practice
  for near-dated single-name options. What matters is using it correctly
  (forward price `F = S*e^((r-q)T)`), not how precisely it's sourced.
- **Dividend yield derived from `dividendRate / spot`**, not yfinance's
  `dividendYield` field directly — that field's units aren't stable
  across yfinance versions. See DEVLOG for the bug this caused.
- **Implied vol solved via Brent's method** (`scipy.optimize.brentq`),
  not bisection — faster convergence, still guaranteed to find a root
  when one exists in the bracket. Returns `None` (not an exception) when
  a market price is unsolvable, since bad individual quotes are routine
  in a live chain and shouldn't kill the whole calculation.
- **European Black-Scholes as an approximation for American options** —
  yfinance equity chains are American-style; the gap is small for
  non-dividend payers and short-dated contracts, larger for dividend
  payers near ex-div and deep ITM puts. Accepted, documented tradeoff.
