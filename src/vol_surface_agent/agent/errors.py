"""
Typed errors for the Milestone 3 graph to branch on explicitly, instead
of the model deciding how to react to an error buried in a tool's
returned string (the Milestone 2 approach — see
tools/langchain_tools.py). Each type maps to a distinct state
transition in agent/graph.py.
"""


class ToolExecutionError(Exception):
    """Base class. error_type is the value stored in graph state so
    conditional edges can route on it without inspecting exception
    classes directly."""

    error_type: str


class BadTickerError(ToolExecutionError):
    """Ticker doesn't exist, isn't valid, or has no data of the
    requested kind (e.g. SPY has no SEC filings). Not retryable —
    retrying with the same ticker won't produce a different result."""

    error_type = "bad_ticker"


class EmptyDataError(ToolExecutionError):
    """The ticker is valid but the specific query returned nothing
    usable (e.g. no near-the-money strikes solved, no matching filing
    chunks). Not retryable for the same query."""

    error_type = "empty_data"


class ApiFailureError(ToolExecutionError):
    """The underlying data source (yfinance, SEC EDGAR) failed to
    respond — a transient network/API issue. Retryable, unlike the
    other two error types."""

    error_type = "api_failure"
