# /// script
# requires-python = ">=3.11"
# dependencies = ["deepagents", "langchain-ollama", "duckdb"]
# ///
"""Benchmark: Deep Agents (LangGraph) tool calling."""

import os
import sys
import time
from typing import Any
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from utils import MODEL, QUESTIONS, SEP, setup_db, make_query_runner, print_summary, append_log

con    = setup_db()
_run_q = make_query_runner(con)


@tool
def run_duckdb_query(sql: str) -> str:
    """Execute a SQL query on DuckDB and return results as a table.

    The database has one table:
    sales(date DATE, product VARCHAR, category VARCHAR,
          quantity INTEGER, price DECIMAL, region VARCHAR).
    Always use this tool to answer data questions.

    Args:
        sql: Valid DuckDB SQL query to execute.
    """
    return _run_q(sql)


class TTFTCallback(BaseCallbackHandler):
    """Records time-to-first-token for the first LLM call per question."""

    def reset(self) -> None:
        self._t_start: float | None = None
        self.ttft:     float | None = None
        self._done:    bool         = False

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        if self._t_start is None:
            self._t_start = time.time()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.ttft is None and self._t_start is not None:
            self.ttft  = time.time() - self._t_start
            self._done = True   # only capture first call


ttft_cb = TTFTCallback()
llm     = ChatOllama(model=MODEL, temperature=0, streaming=True, callbacks=[ttft_cb])
agent   = create_deep_agent(
    model=llm,
    tools=[run_duckdb_query],
    system_prompt=(
        "You are a data analyst. Use run_duckdb_query to answer questions. "
        "Always query the database - never guess numbers."
    ),
)


def run_test(question: str) -> dict:
    print()
    print(SEP)
    print(f"  Q: {question}")
    print(SEP)
    ttft_cb.reset()
    result          = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages        = result.get("messages", [])
    tool_calls_made = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made += 1
                print()
                print(f"  [tool call #{tool_calls_made}] {tc['name']}")
                print(f"  SQL: {tc.get('args', {}).get('sql', '')}")
        if isinstance(msg, ToolMessage):
            print("  Result:")
            print(msg.content)
    final  = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
    answer = final.content.strip() if final else "(no answer)"
    print()
    print("  [answer]")
    print(answer)
    return {"success": bool(final), "tool_calls": tool_calls_made, "ttft": ttft_cb.ttft}


if __name__ == "__main__":
    if os.getenv("BENCH_WARMUP"):
        print("  (packages ready)")
        sys.exit(0)

    results, times, ttfts = [], [], []
    for q in QUESTIONS:
        t0      = time.time()
        r       = run_test(q)
        elapsed = time.time() - t0
        r["time"] = elapsed
        results.append(r)
        times.append(elapsed)
        if r.get("ttft") is not None:
            ttfts.append(r["ttft"])

    print_summary(f"Deep Agents ({MODEL})", results, times, ttfts or None)
    append_log("deep_agents", results, times, ttfts or None)
    con.close()
