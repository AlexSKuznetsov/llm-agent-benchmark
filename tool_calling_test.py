# /// script
# requires-python = ">=3.11"
# dependencies = ["ollama", "duckdb"]
# ///
"""Benchmark: Raw Ollama Python client tool calling."""

import time
import ollama
from utils import MODEL, QUESTIONS, setup_db, make_query_runner, print_summary, append_log

con          = setup_db()
run_query    = make_query_runner(con)

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
    print(f"
{chr(9552)*62}
  Q: {question}
{chr(9552)*62}")
    messages = [
        {"role": "system", "content": "Use run_duckdb_query tool. sales table: date,product,category,quantity,price,region"},
        {"role": "user",   "content": question},
    ]
    tool_calls_made = 0
    for _ in range(max_iterations):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, think=False)
        msg      = response.message
        messages.append(msg)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made += 1
                sql    = tc.function.arguments.get("sql", "")
                result = TOOL_MAP[tc.function.name](sql)
                print(f"
  [tool call #{tool_calls_made}] {tc.function.name}
  SQL: {sql}
  Result:
{result}")
                messages.append({"role": "tool", "content": result})
        else:
            print(f"
  [answer]
{msg.content.strip()}")
            return {"success": True, "tool_calls": tool_calls_made}
    print("
  [!] Max iterations reached.")
    return {"success": False, "tool_calls": tool_calls_made}


if __name__ == "__main__":
    results, times = [], []
    for q in QUESTIONS:
        t0      = time.time()
        r       = run_test(q)
        elapsed = time.time() - t0
        r["time"] = elapsed
        results.append(r)
        times.append(elapsed)
    print_summary(f"Raw Ollama ({MODEL})", results, times)
    append_log("raw_ollama", results, times)
    con.close()
