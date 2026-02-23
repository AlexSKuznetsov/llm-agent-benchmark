import time
import asyncio
import duckdb
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

MODEL   = "openai/qwen3:8b"
OLLAMA  = "http://localhost:11434"

# ── Sample data ───────────────────────────────────────────────────────────────
con = duckdb.connect()
con.execute("""
    CREATE TABLE sales (
        date DATE, product VARCHAR, category VARCHAR,
        quantity INTEGER, price DECIMAL(10,2), region VARCHAR
    )
""")
con.execute("""INSERT INTO sales VALUES
    ('2024-01-01','Laptop','Electronics',5,999.99,'North'),
    ('2024-01-02','Phone','Electronics',12,699.99,'South'),
    ('2024-01-03','Desk','Furniture',3,299.99,'East'),
    ('2024-01-04','Chair','Furniture',8,149.99,'West'),
    ('2024-01-05','Laptop','Electronics',7,999.99,'South'),
    ('2024-01-06','Monitor','Electronics',10,399.99,'North'),
    ('2024-01-07','Keyboard','Electronics',20,79.99,'East'),
    ('2024-02-01','Phone','Electronics',15,699.99,'North'),
    ('2024-02-02','Desk','Furniture',5,299.99,'South'),
    ('2024-02-03','Chair','Furniture',12,149.99,'North'),
    ('2024-02-04','Laptop','Electronics',9,999.99,'West'),
    ('2024-02-05','Monitor','Electronics',6,399.99,'East'),
    ('2024-03-01','Phone','Electronics',18,699.99,'West'),
    ('2024-03-02','Keyboard','Electronics',25,79.99,'South'),
    ('2024-03-03','Desk','Furniture',4,299.99,'North'),
    ('2024-03-04','Chair','Furniture',15,149.99,'East'),
    ('2024-03-05','Laptop','Electronics',11,999.99,'South'),
    ('2024-03-06','Monitor','Electronics',8,399.99,'West')
""")

# ── Tool ──────────────────────────────────────────────────────────────────────
def run_duckdb_query(sql: str) -> str:
    """Execute a SQL query on DuckDB and return the results as a table.
    The database has a 'sales' table with columns: date DATE, product VARCHAR,
    category VARCHAR, quantity INTEGER, price DECIMAL, region VARCHAR.
    Always use this tool to answer data questions — never guess numbers.

    Args:
        sql: Valid DuckDB SQL query to execute.

    Returns:
        Query results as a formatted table string.
    """
    try:
        result = con.execute(sql)
        cols = [d[0] for d in result.description]
        rows = result.fetchall()
        if not rows:
            return "No results."
        widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines = [fmt.format(*cols), "  ".join("-"*w for w in widths)]
        lines += [fmt.format(*[str(v) for v in row]) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"


# ── Agent ─────────────────────────────────────────────────────────────────────
import os
os.environ["OPENAI_API_BASE"] = f"{OLLAMA}/v1"
os.environ["OPENAI_API_KEY"]  = "ollama"

agent = Agent(
    model=LiteLlm(model=MODEL),
    name="data_analyst",
    description="A data analyst that queries a DuckDB sales database.",
    instruction=(
        "/no_think\n"
        "You are a data analyst. Use the run_duckdb_query tool to answer "
        "questions. Always query the database — never guess numbers."
    ),
    tools=[run_duckdb_query],
)

# ── Questions ─────────────────────────────────────────────────────────────────
QUESTIONS = [
    "What is the total revenue per category?",
    "Which region had the highest total revenue?",
    "Show monthly revenue for each month.",
    "What are the top 3 best-selling products by quantity?",
    "Which product has the best revenue-to-quantity ratio?",
]


# ── Runner ────────────────────────────────────────────────────────────────────
async def run_test(question: str, runner: InMemoryRunner) -> dict:
    print(f"\n{'═'*62}")
    print(f"  Q: {question}")
    print("═"*62)

    tool_calls_made = 0
    final_answer    = ""

    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=question)]
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls_made += 1
                    fc = part.function_call
                    sql = fc.args.get("sql", "")
                    print(f"\n  [tool call #{tool_calls_made}] {fc.name}")
                    print(f"  SQL: {sql}")
                    result = run_duckdb_query(sql)
                    print(f"  Result:\n{result}")
                elif hasattr(part, "text") and part.text and event.is_final_response():
                    final_answer = part.text.strip()

    if final_answer:
        print(f"\n  [answer]\n{final_answer}")
        return {"success": True, "tool_calls": tool_calls_made}

    print("\n  [!] No final answer.")
    return {"success": False, "tool_calls": tool_calls_made}


async def main():
    runner  = InMemoryRunner(agent=agent, app_name="bench")
    results = []
    times   = []

    for q in QUESTIONS:
        # fresh session per question
        await runner.session_service.create_session(
            app_name="bench", user_id="user", session_id="session"
        )
        t0      = time.time()
        r       = await run_test(q, runner)
        elapsed = time.time() - t0
        r["time"] = elapsed
        results.append(r)
        times.append(elapsed)

        # delete session so next question starts fresh
        await runner.session_service.delete_session(
            app_name="bench", user_id="user", session_id="session"
        )

    print(f"\n{'═'*62}")
    print(f"  SUMMARY  (qwen3:8b via Google ADK + LiteLLM)")
    print("═"*62)
    passed    = sum(1 for r in results if r["success"])
    avg_calls = sum(r["tool_calls"] for r in results) / len(results)
    avg_time  = sum(times) / len(times)
    print(f"  Passed           : {passed}/{len(QUESTIONS)}")
    print(f"  Avg tool calls/Q : {avg_calls:.1f}")
    print(f"  Avg time/Q       : {avg_time:.2f}s")
    for i, (q, r) in enumerate(zip(QUESTIONS, results), 1):
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] Q{i} — {r['tool_calls']} call(s)  {r['time']:.1f}s  |  {q[:45]}")
    print()
    con.close()


if __name__ == "__main__":
    asyncio.run(main())
