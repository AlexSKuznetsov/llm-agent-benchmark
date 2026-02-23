# /// script
# requires-python = ">=3.11"
# dependencies = ["google-adk[extensions]", "duckdb"]
# ///
"""Benchmark: Google ADK + LiteLLM tool calling."""

import os
import sys
import time
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types
from utils import MODEL, QUESTIONS, SEP, setup_db, make_query_runner, print_summary, append_log

ADK_MODEL = f"openai/{MODEL}"
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY",  "ollama")

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
    model=LiteLlm(model=ADK_MODEL),
    name="data_analyst",
    description="A data analyst that queries a DuckDB sales database.",
    instruction=(
        "/no_think\n"
        "You are a data analyst. Use run_duckdb_query to answer questions. "
        "Always query the database - never guess numbers."
    ),
    tools=[run_duckdb_query],
)


async def run_test(question: str, runner: InMemoryRunner) -> dict:
    print()
    print(SEP)
    print(f"  Q: {question}")
    print(SEP)
    tool_calls_made = 0
    final_answer    = ""
    ttft            = None
    t_start         = time.time()

    async for event in runner.run_async(
        user_id="user", session_id="session",
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
    ):
        # Capture TTFT on first event that carries content
        if ttft is None and event.content and event.content.parts:
            ttft = time.time() - t_start

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls_made += 1
                    fc  = part.function_call
                    sql = fc.args.get("sql", "")
                    res = _run_q(sql)
                    print()
                    print(f"  [tool call #{tool_calls_made}] {fc.name}")
                    print(f"  SQL: {sql}")
                    print("  Result:")
                    print(res)
                elif hasattr(part, "text") and part.text and event.is_final_response():
                    final_answer = part.text.strip()

    if final_answer:
        print()
        print("  [answer]")
        print(final_answer)
    return {"success": bool(final_answer), "tool_calls": tool_calls_made, "ttft": ttft}


async def main():
    runner         = InMemoryRunner(agent=agent, app_name="bench")
    results, times, ttfts = [], [], []
    for q in QUESTIONS:
        await runner.session_service.create_session(
            app_name="bench", user_id="user", session_id="session"
        )
        t0      = time.time()
        r       = await run_test(q, runner)
        elapsed = time.time() - t0
        r["time"] = elapsed
        results.append(r)
        times.append(elapsed)
        if r.get("ttft") is not None:
            ttfts.append(r["ttft"])
        await runner.session_service.delete_session(
            app_name="bench", user_id="user", session_id="session"
        )
    print_summary(f"Google ADK ({MODEL})", results, times, ttfts or None)
    append_log("google_adk", results, times, ttfts or None)
    con.close()


if __name__ == "__main__":
    if os.getenv("BENCH_WARMUP"):
        print("  (packages ready)")
        sys.exit(0)

    asyncio.run(main())
