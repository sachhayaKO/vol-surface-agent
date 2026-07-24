"""
Milestone 4 eval runner. Scores the Milestone 3 graph agent
(agent/graph.py) against evals/golden_dataset.json using two kinds of
checks per question:

- Deterministic: was the right tool called, with the right ticker; did
  a required keyword show up (e.g. refusal language for an out-of-scope
  ticker); did a forbidden pattern show up (e.g. a fabricated number).
- LLM-as-judge: a rubric describing what a *good* answer does, scored by
  a separate Claude call, for things too fuzzy for keyword matching
  (does the answer actually synthesize across tools, not just report
  numbers back).

Rerun this after any change to the agent, tools, or system prompt.

Usage:
    python -m evals.run_eval [--limit N] [--category NAME]
"""

import argparse
import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from vol_surface_agent.agent.graph import SYSTEM_PROMPT, build_agent

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "last_run_results.json"
JUDGE_MODEL = "claude-sonnet-5"

JUDGE_PROMPT = """You are grading an AI options-volatility assistant's answer.

Question(s) asked, in order: {turns}

Raw tool output the assistant actually retrieved during this conversation \
(this is the ground truth — anything in the final answer should be \
traceable to this, market data conventions, or basic arithmetic on it; \
do NOT flag specific numbers/dates as suspicious just because they look \
unfamiliar or future-dated to you — this agent works with live current \
market data newer than your training data, so dates and figures you \
don't recognize can still be genuine):
{tool_evidence}

Assistant's final answer:
{answer}

Grading rubric: {rubric}

Respond with exactly one line: "PASS" or "FAIL", then a one-sentence reason on the next line."""


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def extract_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def run_turns(agent, turns: list[str], thread_id: str) -> tuple[str, list[tuple[str, dict]], list[str]]:
    """
    Run each turn in sequence on a fresh thread. Returns the final
    turn's answer text, every tool call made across all turns, and the
    raw content every tool actually returned (needed so the judge can
    check groundedness against real retrieved evidence instead of
    guessing from the final answer's plausibility alone — see
    Sachin's Notebook, Milestone 4 entry, for why this matters).
    """
    config = {"configurable": {"thread_id": thread_id}}
    tool_calls = []
    tool_results = []
    result = None

    for i, turn in enumerate(turns):
        messages = [HumanMessage(content=turn)]
        if i == 0:
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        result = agent.invoke({"messages": messages}, config)
        for m in result["messages"]:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                tool_calls.extend((tc["name"], tc["args"]) for tc in m.tool_calls)
            if isinstance(m, ToolMessage):
                tool_results.append(str(m.content))

    return extract_text(result["messages"][-1]), tool_calls, tool_results


def run_deterministic_checks(case: dict, final_text: str, tool_calls: list[tuple[str, dict]]) -> list[str]:
    checks = case.get("checks", {})
    failures = []
    called_names = {name for name, _ in tool_calls}
    text_lower = final_text.lower()

    expected_tools = set(checks.get("expected_tools", []))
    if expected_tools and not expected_tools.issubset(called_names):
        failures.append(f"expected tools {expected_tools}, got {called_names or 'none'}")

    forbidden_tools = set(checks.get("forbidden_tools", []))
    hit_forbidden = forbidden_tools & called_names
    if hit_forbidden:
        failures.append(f"forbidden tools called: {hit_forbidden}")

    for tool_name, expected_args in checks.get("expected_args", {}).items():
        matching_calls = [args for name, args in tool_calls if name == tool_name]
        if not matching_calls:
            failures.append(f"{tool_name} was never called (can't check its args)")
            continue
        for key, expected_val in expected_args.items():
            if not any(str(args.get(key, "")).upper() == str(expected_val).upper() for args in matching_calls):
                failures.append(f"{tool_name} never called with {key}={expected_val}")

    required_any = checks.get("required_keywords_any", [])
    if required_any and not any(kw.lower() in text_lower for kw in required_any):
        failures.append(f"none of required keywords {required_any} found in answer")

    forbidden_kw = [kw for kw in checks.get("forbidden_keywords", []) if kw.lower() in text_lower]
    if forbidden_kw:
        failures.append(f"forbidden keywords present: {forbidden_kw}")

    max_calls = checks.get("max_tool_calls")
    if max_calls is not None and len(tool_calls) > max_calls:
        failures.append(f"expected at most {max_calls} tool calls, got {len(tool_calls)}")

    return failures


def run_judge(judge_model, case: dict, final_text: str, tool_results: list[str]) -> dict | None:
    rubric = case.get("judge_rubric")
    if not rubric:
        return None
    evidence = "\n---\n".join(tool_results) if tool_results else "(no tools were called this conversation)"
    prompt = JUDGE_PROMPT.format(turns=case["turns"], answer=final_text, rubric=rubric, tool_evidence=evidence)
    response = judge_model.invoke(prompt)
    text = extract_text(response)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    verdict = lines[0].upper() if lines else "UNKNOWN"
    reason = lines[1] if len(lines) > 1 else ""
    return {"verdict": verdict, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    dataset = load_dataset()
    if args.category:
        dataset = [c for c in dataset if c["category"] == args.category]
    if args.limit:
        dataset = dataset[: args.limit]

    agent = build_agent()
    judge_model = ChatAnthropic(model=JUDGE_MODEL)

    results = []
    for case in dataset:
        print(f"Running {case['id']} ({case['category']})...")
        try:
            final_text, tool_calls, tool_results = run_turns(agent, case["turns"], f"eval-{case['id']}")
        except Exception as e:  # noqa: BLE001 - a live agent run can fail in many ways; capture and keep going
            print(f"  ERROR: {e}")
            results.append({"id": case["id"], "category": case["category"], "passed": False, "error": str(e)})
            continue

        det_failures = run_deterministic_checks(case, final_text, tool_calls)
        judge_result = run_judge(judge_model, case, final_text, tool_results)
        passed = not det_failures and (judge_result is None or judge_result["verdict"] == "PASS")

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": passed,
                "deterministic_failures": det_failures,
                "judge": judge_result,
                "tool_calls": tool_calls,
                "answer_preview": final_text[:300],
            }
        )
        status = "PASS" if passed else "FAIL"
        detail = det_failures or (judge_result["reason"] if judge_result else "")
        print(f"  {status}" + (f" — {detail}" if detail else ""))

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}\n{passed_count}/{total} passed\n{'='*60}")

    by_category: dict[str, list[int]] = {}
    for r in results:
        bucket = by_category.setdefault(r["category"], [0, 0])
        bucket[1] += 1
        bucket[0] += int(r["passed"])
    for cat, (p, t) in sorted(by_category.items()):
        print(f"  {cat}: {p}/{t}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
