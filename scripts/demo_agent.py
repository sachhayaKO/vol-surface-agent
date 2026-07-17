"""
Manual multi-turn test of the Milestone 2 agent. Not an automated eval
(that's Milestone 4) — a script to run a handful of real questions
through the agent and read the output, the same "call it and look"
discipline as scripts/demo_iv.py.

Requires ANTHROPIC_API_KEY in the environment.

Usage:
    python scripts/demo_agent.py
"""

from vol_surface_agent.agent.simple_agent import build_agent


def extract_text(message) -> str:
    """
    Claude's extended-thinking responses come back as a list of content
    blocks (thinking + text), not a plain string — printing
    message.content directly dumps raw block dicts (including the
    thinking signature blob) instead of the answer. Pull out just the
    text blocks.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


QUESTIONS = [
    "What does AAPL's implied vol surface look like right now?",
    "How does that compare to AAPL's realized vol? Does it look rich or cheap?",
    "Is there anything in AAPL's recent filings that would explain the current vol level?",
    "What about SPY — can you show me its options and tell me about its recent earnings?",
]


def main() -> None:
    agent = build_agent()
    thread_config = {"configurable": {"thread_id": "demo-thread-1"}}

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}\nTurn {i}: {question}\n{'='*70}")
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, thread_config)
        final_message = result["messages"][-1]
        print(f"\nAgent: {extract_text(final_message)}")


if __name__ == "__main__":
    main()
