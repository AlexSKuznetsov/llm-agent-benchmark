# /// script
# requires-python = ">=3.11"
# dependencies = ["deepagents", "langchain-ollama", "duckdb"]
# ///
"""Benchmark: Deep Agents (LangGraph) tool calling."""

import time
from langchain_ollama import ChatOllama
from langchain.tools import tool
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


llm   = ChatOllama(model=MODEL, temperature=0)
agent = create_deep_agent(
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
            print(f"  Result:")
            print(msg.content)
    final  = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
    answer = final.content.strip() if final else "(no answer)"
    print()
    print("  [answer]")
    print(answer)
    return {"success": bool(final), "tool_calls": tool_calls_made}


if __name__ == "__main__":
    results, times = [], []
    for q in QUESTIONS:
        t0      = time.time()
        r       = run_test(q)
        elapsed = time.time() - t0
        r["time"] = elapsed
        results.append(r)
        times.append(elapsed)
    print_summary(f"Deep Agents ({MODEL})", results, times)
    append_log("deep_agents", results, times)
    con.close()
