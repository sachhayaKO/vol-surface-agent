"""
Manual test of the Milestone 3 custom StateGraph agent — same "call it
and read the output" discipline as the other demo scripts, but this one
deliberately includes a bad ticker to exercise the explicit error-
handling path (execute_tool -> handle_error -> reason), not just the
happy path already proven in Milestone 2.

Requires ANTHROPIC_API_KEY in the environment (.env is loaded
automatically).

Usage:
    python scripts/demo_graph_agent.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from vol_surface_agent.agent.graph import SYSTEM_PROMPT, build_agent, execute_tool

QUESTIONS = [
    "What does NVDA's implied vol look like right now?",
    "What's the realized vol for ZZZFAKETICKER — how does it compare?",
    "Never mind that one — how does NVDA's implied vol compare to its realized vol?",
]


def extract_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def demo_conversation() -> None:
    agent = build_agent()
    thread_config = {"configurable": {"thread_id": "graph-demo-thread-1"}}

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}\nTurn {i}: {question}\n{'='*70}")

        messages = [HumanMessage(content=question)]
        if i == 1:
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        result = agent.invoke({"messages": messages}, thread_config)
        final_message = result["messages"][-1]
        print(f"\nAgent: {extract_text(final_message)}")
        print(f"\n[state check] last_error={result.get('last_error')}  retry_count={result.get('retry_count')}")


def demo_forced_error_path() -> None:
    """
    Turn 2 above shows the model declining a fake ticker on its own,
    guided by the system prompt — good behavior, but it means the
    graph's error-handling machinery (execute_tool -> handle_error)
    never actually runs during that conversation. This calls
    execute_tool directly with a tool call the model *would* have made
    if it had tried, to prove the live path (real failed network call
    -> typed error -> classified state) actually works, independent of
    whether the model happens to attempt it in a given conversation.
    """
    print(f"\n{'='*70}\nForced error path: execute_tool with a real bad ticker\n{'='*70}")
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_realized_vol", "args": {"ticker": "ZZZFAKETICKER"}, "id": "call_1"}],
            )
        ]
    }
    result = execute_tool(state)
    print(f"\nlast_error: {result['last_error']}")
    print(f"tool message: {result['messages'][0].content}")


if __name__ == "__main__":
    demo_conversation()
    demo_forced_error_path()
