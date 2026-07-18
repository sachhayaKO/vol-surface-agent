"""
Unit tests for the Milestone 3 graph's routing and error-handling logic.
No network or LLM calls — these test the state transitions in isolation,
same philosophy as the rest of the suite. execute_tool's malformed-args
path is exercised directly (invalid tool args fail Pydantic validation
before any network call happens), but its success/live-error paths are
exercised manually via scripts/demo_graph_agent.py instead, since those
need a real ticker/network round-trip.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from vol_surface_agent.agent.graph import (
    MAX_RETRIES,
    execute_tool,
    handle_error,
    route_after_error,
    route_after_reason,
    route_after_tool,
)


def _ai_message_with_tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call_1"}],
    )


def test_route_after_reason_goes_to_execute_tool_when_tool_calls_present():
    state = {"messages": [_ai_message_with_tool_call("get_realized_vol", {"ticker": "AAPL"})]}
    assert route_after_reason(state) == "execute_tool"


def test_route_after_reason_ends_when_final_answer():
    state = {"messages": [AIMessage(content="Here's your answer.")]}
    assert route_after_reason(state) == END


def test_route_after_tool_goes_to_handle_error_when_error_set():
    assert route_after_tool({"last_error": "bad_ticker"}) == "handle_error"


def test_route_after_tool_goes_to_reason_when_no_error():
    assert route_after_tool({"last_error": None}) == "reason"


def test_route_after_error_retries_under_cap():
    assert route_after_error({"retry_count": 1}) == "reason"


def test_route_after_error_ends_over_cap():
    assert route_after_error({"retry_count": MAX_RETRIES + 1}) == END


def test_handle_error_injects_system_note_and_increments_retry_under_cap():
    state = {"last_error": "malformed_args", "retry_count": 0}
    result = handle_error(state)

    assert result["retry_count"] == 1
    assert result["last_error"] is None
    note = result["messages"][0]
    assert isinstance(note, HumanMessage)
    assert note.content.startswith("[System note]")


def test_handle_error_gives_up_deterministically_over_cap():
    state = {"last_error": "api_failure", "retry_count": MAX_RETRIES}
    result = handle_error(state)

    assert result["retry_count"] == MAX_RETRIES + 1
    give_up_message = result["messages"][0]
    assert "stopping" in give_up_message.content.lower()


def test_execute_tool_flags_malformed_args_for_missing_required_field():
    state = {"messages": [_ai_message_with_tool_call("get_realized_vol", {})]}  # missing "ticker"
    result = execute_tool(state)

    assert result["last_error"] == "malformed_args"
    assert len(result["messages"]) == 1  # every tool call still gets a ToolMessage


def test_execute_tool_flags_malformed_args_for_unknown_tool_name():
    state = {"messages": [_ai_message_with_tool_call("not_a_real_tool", {"ticker": "AAPL"})]}
    result = execute_tool(state)

    assert result["last_error"] == "malformed_args"
    assert "Unknown tool" in result["messages"][0].content
