"""
Milestone 3 agent: a custom LangGraph StateGraph replacing create_agent's
default loop, so error handling is an explicit, inspectable state
transition instead of prose the model has to interpret for itself.

See docs/ARCHITECTURE.md and the Design and Architecture / Agent_Concepts
notes in the Obsidian vault for the tradeoffs vs create_agent (Milestone 2).

Graph shape:

    reason --(tool call requested)--> execute_tool --(failed)--> handle_error
      ^                                    |                          |
      |                              (succeeded)                (retryable)
      |                                    |                          |
      +------------------------------------+--------------------------+
      |
      (no tool call: final answer)
      v
     END                                                    (retries exhausted)
                                                                        |
                                                                        v
                                                                       END
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from vol_surface_agent.agent.errors import ToolExecutionError
from vol_surface_agent.agent.graph_tools import GRAPH_TOOLS

load_dotenv()

MAX_RETRIES = 2

ErrorType = Literal["api_failure", "bad_ticker", "empty_data", "malformed_args"] | None

TOOLS_BY_NAME = {t.name: t for t in GRAPH_TOOLS}

SYSTEM_PROMPT = """\
You are an options volatility analyst assistant. You have tools to fetch \
live implied volatility (from options prices), realized volatility (from \
historical price data), and search recent earnings releases / 10-K risk \
factor text for AAPL and NVDA (SPY is an ETF and has no such filings).

Coverage is currently limited to AAPL, SPY, and NVDA for options/realized \
vol, and AAPL/NVDA for filing search. If asked about any other ticker, \
say so rather than guessing.

When asked whether volatility looks "rich" or "cheap", compare implied \
vol against realized vol at a comparable horizon, and check whether \
there's a known reason (from filings/earnings context) that implied vol \
should be elevated before concluding it's mispriced. Be specific about \
which expiry/strike or which realized-vol window/estimator you're citing.

You may occasionally see a message starting with "[System note]" — this \
is not from the user. It's guidance about a tool call that just failed. \
Follow its instructions (retry with corrected input, or tell the user \
plainly) rather than treating it as a new user request.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_error: ErrorType
    retry_count: int


ERROR_GUIDANCE = {
    "malformed_args": (
        "Your last tool call had invalid arguments. Check the tool's expected "
        "argument names and types and try again with corrected arguments."
    ),
    "bad_ticker": (
        "The ticker in your last tool call isn't valid or supported. Don't "
        "retry with the same ticker — tell the user plainly that it isn't supported."
    ),
    "empty_data": (
        "That request returned no usable data. Don't retry the identical "
        "request — tell the user plainly that no data was available."
    ),
    "api_failure": (
        "The data source failed to respond, which may be transient. You may "
        "retry once, but if it keeps failing, tell the user the service is "
        "temporarily unavailable."
    ),
}


def reason(state: AgentState, model) -> dict:
    """Call the model with the current conversation. It either responds
    with tool call(s) or a final answer — route_after_reason decides
    which happened."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def execute_tool(state: AgentState) -> dict:
    """
    Run every tool call the model just requested. Every tool call gets
    a ToolMessage response no matter what (required by the Anthropic/
    LangChain tool-calling protocol — a tool_call without a matching
    result breaks the next model call), even on failure; failures also
    set last_error so route_after_tool can send execution to
    handle_error instead of straight back to reason.

    If multiple tool calls happen in one turn and more than one fails,
    the first failure's error type wins — good enough for this
    project's scope (one retry loop, not per-call error tracking).
    """
    last_message = state["messages"][-1]
    tool_messages = []
    detected_error: ErrorType = None

    for call in last_message.tool_calls:
        tool_fn = TOOLS_BY_NAME.get(call["name"])

        if tool_fn is None:
            tool_messages.append(
                ToolMessage(content=f"Unknown tool '{call['name']}'.", tool_call_id=call["id"])
            )
            detected_error = detected_error or "malformed_args"
            continue

        try:
            result = tool_fn.invoke(call["args"])
        except ValidationError as e:
            tool_messages.append(
                ToolMessage(content=f"Invalid arguments for {call['name']}: {e}", tool_call_id=call["id"])
            )
            detected_error = detected_error or "malformed_args"
        except ToolExecutionError as e:
            tool_messages.append(ToolMessage(content=str(e), tool_call_id=call["id"]))
            detected_error = detected_error or e.error_type
        else:
            tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    if detected_error is None:
        return {"messages": tool_messages, "last_error": None, "retry_count": 0}
    return {"messages": tool_messages, "last_error": detected_error}


def handle_error(state: AgentState) -> dict:
    """
    Decide how to react to the error execute_tool just classified.
    Under the retry cap: inject a "[System note]" message nudging the
    model toward the right reaction (retry vs. tell the user) and route
    back to reason. Over the cap: stop deterministically with a plain
    message instead of risking another failed attempt.
    """
    error = state["last_error"]
    retry_count = state.get("retry_count", 0) + 1

    if retry_count > MAX_RETRIES:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"I'm stopping after repeated {error.replace('_', ' ')} errors and "
                        f"couldn't complete this request. Please try rephrasing or double-check "
                        f"the ticker/request."
                    )
                )
            ],
            "retry_count": retry_count,
            "last_error": None,
        }

    return {
        "messages": [HumanMessage(content=f"[System note] {ERROR_GUIDANCE[error]}")],
        "retry_count": retry_count,
        "last_error": None,
    }


def route_after_reason(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "execute_tool"
    return END


def route_after_tool(state: AgentState) -> str:
    return "handle_error" if state.get("last_error") else "reason"


def route_after_error(state: AgentState) -> str:
    return END if state.get("retry_count", 0) > MAX_RETRIES else "reason"


def build_agent(checkpointer: InMemorySaver | None = None):
    """Build and compile the Milestone 3 graph."""
    model = ChatAnthropic(model="claude-sonnet-5").bind_tools(GRAPH_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("reason", lambda state: reason(state, model))
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", route_after_reason, {"execute_tool": "execute_tool", END: END})
    graph.add_conditional_edges("execute_tool", route_after_tool, {"reason": "reason", "handle_error": "handle_error"})
    graph.add_conditional_edges("handle_error", route_after_error, {"reason": "reason", END: END})

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
