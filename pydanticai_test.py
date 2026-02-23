# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic-ai", "duckdb"]
# ///
"""Benchmark: Pydantic AI (Ollama provider) tool calling."""

import json
import asyncio
import os
import sys
import time
from pydantic_ai import Agent
from utils import MODEL, QUESTIONS, SEP, setup_db, make_query_runner, print_summary, append_log

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")

con    = setup_db()
_run_q = make_query_runner(con)


def run_duckdb_query(sql: str) -> str:
    """Execute a SQL query on DuckDB and return results as a table.

    The database has one table:
    sales(date DATE, product VARCHAR, category VARCHAR,
          quantity INTEGER, price DECIMAL, region VARCHAR).
    Always use this tool to answer data questions.

    Args:
        sql: Valid DuckDB SQL query to execute.

    Returns:
        Query results formatted as a plain-text table.
    """
    return _run_q(sql)


agent = Agent(
    f"ollama:{MODEL}",
    instructions=(
        "You are a data analyst. Use run_duckdb_query to answer questions. "
        "Always query the database - never guess numbers."
    ),
    tools=[run_duckdb_query],
)


def _parse_tool_args(raw_args) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    args_dict = getattr(raw_args, "args_dict", None)
    if isinstance(args_dict, dict):
        return args_dict
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


async def run_test(question: str) -> dict:
    print()
    print(SEP)
    print(f"  Q: {question}")
    print(SEP)

    t0 = time.time()
    ttft = None
    result = None
    async for event in agent.run_stream_events(question):
        if ttft is None and event.__class__.__name__ != "AgentRunResultEvent":
            ttft = time.time() - t0
        if hasattr(event, "result"):
            result = event.result
    elapsed = time.time() - t0

    if result is None:
        return {"success": False, "tool_calls": 0, "ttft": ttft, "time": elapsed}

    tool_calls_made = 0
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            kind = getattr(part, "part_kind", "")
            if kind == "tool-call":
                tool_calls_made += 1
                name = getattr(part, "tool_name", "run_duckdb_query")
                args = _parse_tool_args(getattr(part, "args", {}))
                sql = args.get("sql", "")
                print()
                print(f"  [tool call #{tool_calls_made}] {name}")
                print(f"  SQL: {sql}")
            elif kind == "tool-return":
                print("  Result:")
                print(getattr(part, "content", ""))

    answer = str(result.output).strip()
    print()
    print("  [answer]")
    print(answer)
    return {
        "success": bool(answer),
        "tool_calls": tool_calls_made,
        "ttft": ttft,
        "time": elapsed,
    }


async def main() -> None:
    results, times, ttfts = [], [], []
    for q in QUESTIONS:
        r = await run_test(q)
        results.append(r)
        times.append(r["time"])
        if r.get("ttft") is not None:
            ttfts.append(r["ttft"])

    print_summary(f"Pydantic AI ({MODEL})", results, times, ttfts or None)
    append_log("pydantic_ai", results, times, ttfts or None)
    con.close()


if __name__ == "__main__":
    if os.getenv("BENCH_WARMUP"):
        print("  (packages ready)")
        sys.exit(0)

    asyncio.run(main())
