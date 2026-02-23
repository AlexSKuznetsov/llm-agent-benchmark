# /// script
# requires-python = ">=3.11"
# dependencies = ["ollama", "duckdb"]
# ///
"""Benchmark: Raw Ollama Python client tool calling."""

import os
import sys
import time
import ollama
from utils import MODEL, QUESTIONS, SEP, setup_db, make_query_runner, print_summary, append_log

con       = setup_db()
run_query = make_query_runner(con)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_duckdb_query",
            "description": run_query.__doc__,
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "DuckDB SQL to execute"}},
                "required": ["sql"],
            },
        },
    }
]

TOOL_MAP = {"run_duckdb_query": run_query}


def run_test(question: str, max_iterations: int = 6) -> dict:
    print()
    print(SEP)
    print(f"  Q: {question}")
    print(SEP)
    messages = [
        {"role": "system", "content": "Use run_duckdb_query tool. sales table: date,product,category,quantity,price,region"},
        {"role": "user",   "content": question},
    ]
    tool_calls_made = 0
    ttft            = None          # captured on first LLM call only

    for _ in range(max_iterations):
        t_call = time.time()
        stream = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, think=False, stream=True)

        full_response = None
        for chunk in stream:
            if ttft is None:            # first token of first call
                ttft = time.time() - t_call
            full_response = chunk       # keep updating; last chunk is complete

        msg = full_response.message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made += 1
                sql    = tc.function.arguments.get("sql", "")
                result = TOOL_MAP[tc.function.name](sql)
                print()
                print(f"  [tool call #{tool_calls_made}] {tc.function.name}")
                print(f"  SQL: {sql}")
                print("  Result:")
                print(result)
                messages.append({"role": "tool", "content": result})
        else:
            print()
            print("  [answer]")
            print(msg.content.strip())
            return {"success": True, "tool_calls": tool_calls_made, "ttft": ttft}

    print()
    print("  [!] Max iterations reached.")
    return {"success": False, "tool_calls": tool_calls_made, "ttft": ttft}


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

    print_summary(f"Raw Ollama ({MODEL})", results, times, ttfts or None)
    append_log("raw_ollama", results, times, ttfts or None)
    con.close()
