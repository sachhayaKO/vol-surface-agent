# vol-surface-agent

An agent for options volatility analysis. It pulls a live options chain,
computes implied vol and greeks, retrieves grounding context from recent
earnings calls / 10-Ks, and answers questions about skew and whether
volatility looks rich or cheap versus realized vol.

Built with LangGraph/LangChain, dropping down to a custom `StateGraph`
rather than staying at the high-level agent API — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the
reasoning behind that tradeoff.

## Status

Early — the vol math foundation (options chain retrieval, Black-Scholes,
implied vol) is built and tested. The agent layer, retrieval, and eval
suite are not yet built. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the build plan and current status of each piece.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

Run the implied vol calculation against a live chain:

```bash
python scripts/demo_iv.py AAPL
```

Run tests:

```bash
pytest
```

## Layout

```
src/vol_surface_agent/
  tools/       options chain retrieval, Black-Scholes, implied vol
  agent/       LangChain tools + LangGraph state machine (not yet built)
  ingestion/   earnings/10-K retrieval for the Chroma retriever tool (not yet built)
scripts/       manual run/demo entrypoints
tests/         unit tests
evals/         golden dataset + scoring scripts (not yet built)
docs/          architecture and build plan
```
