"""
Milestone 2 agent: LangChain tools wired into create_agent's default
ReAct-style loop, with an in-memory checkpointer for multi-turn state.

This intentionally uses the framework's default agent loop rather than a
custom graph — that's Milestone 3. See docs/ARCHITECTURE.md for the
tradeoffs between the two.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from vol_surface_agent.tools.langchain_tools import (
    get_implied_vol_surface,
    get_realized_vol,
    search_earnings_and_filings,
)

load_dotenv()

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
should be elevated (e.g. an upcoming or recent earnings report) before \
concluding it's mispriced. Be specific about which expiry/strike or \
which realized-vol window and estimator you're citing — don't answer in \
vague terms when the tools gave you specific numbers.
"""


def build_agent(checkpointer: InMemorySaver | None = None):
    """Build the Milestone 2 agent. Pass a shared checkpointer if the
    caller wants to manage its lifecycle (e.g. tests); otherwise a new
    one is created."""
    model = ChatAnthropic(model="claude-sonnet-5")
    tools = [get_implied_vol_surface, get_realized_vol, search_earnings_and_filings]

    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )
