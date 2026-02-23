import duckdb
from langchain_ollama import ChatOllama
from langchain.tools import tool
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage

MODEL = "qwen3:8b"

# ── Sample data ───────────────────────────────────────────────────────────────
con = duckdb.connect()
con.execute("""
    CREATE TABLE sales (
        date     DATE,
        product  VARCHAR,
        category VARCHAR,
        quantity INTEGER,
        price    DECIMAL(10, 2),
        region   VARCHAR
    )
""")
con.execute("""
    INSERT INTO sales VALUES
        ('2024-01-01', 'Laptop',   'Electronics', 5,  999.99, 'North'),
        ('2024-01-02', 'Phone',    'Electronics', 12, 699.99, 'South'),
        ('2024-01-03', 'Desk',     'Furniture',   3,  299.99, 'East'),
        ('2024-01-04', 'Chair',    'Furniture',   8,  149.99, 'West'),
        ('2024-01-05', 'Laptop',   'Electronics', 7,  999.99, 'South'),
        ('2024-01-06', 'Monitor',  'Electronics', 10, 399.99, 'North'),
        ('2024-01-07', 'Keyboard', 'Electronics', 20,  79.99, 'East'),
        ('2024-02-01', 'Phone',    'Electronics', 15, 699.99, 'North'),
        ('2024-02-02', 'Desk',     'Furniture',   5,  299.99, 'South'),
        ('2024-02-03', 'Chair',    'Furniture',   12, 149.99, 'North'),
        ('2024-02-04', 'Laptop',   'Electronics', 9,  999.99, 'West'),
        ('2024-02-05', 'Monitor',  'Electronics', 6,  399.99, 'East'),
        ('2024-03-01', 'Phone',    'Electronics', 18, 699.99, 'West'),
        ('2024-03-02', 'Keyboard', 'Electronics', 25,  79.99, 'South'),
        ('2024-03-03', 'Desk',     'Furniture',   4,  299.99, 'North'),
        ('2024-03-04', 'Chair',    'Furniture',   15, 149.99, 'East'),
        ('2024-03-05', 'Laptop',   'Electronics', 11, 999.99, 'South'),
        ('2024-03-06', 'Monitor',  'Electronics', 8,  399.99, 'West')
""")


# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def run_duckdb_query(sql: str) -> str:
    """Execute a SQL query on DuckDB and return results as a table.
    The database has one table: sales(date DATE, product VARCHAR,
    category VARCHAR, quantity INTEGER, price DECIMAL, region VARCHAR).
    Always use this tool to answer data questions — do not guess numbers.
    """
    try:
        result = con.execute(sql)
        cols = [desc[0] for desc in result.description]
        rows = result.fetchall()
        if not rows:
            return "Query returned no results."
        col_widths = [
            max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
            for i, c in enumerate(cols)
        ]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        lines = [fmt.format(*cols), "  ".join("-" * w for w in col_widths)]
        lines += [fmt.format(*[str(v) for v in row]) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"


# ── Agent setup ───────────────────────────────────────────────────────────────
llm = ChatOllama(model=MODEL, temperature=0)

agent = create_deep_agent(
    model=llm,
    tools=[run_duckdb_query],
    system_prompt=(
        "You are a data analyst. Use the run_duckdb_query tool to answer "
        "questions. Always query the database — do not guess the numbers."
    ),
)

# ── Questions ─────────────────────────────────────────────────────────────────
QUESTIONS = [
    "What is the total revenue per category?",
    "Which region had the highest total revenue?",
    "Show monthly revenue for each month.",
    "What are the top 3 best-selling products by quantity?",
    "Which product has the best revenue-to-quantity ratio?",
]


# ── Run tests ─────────────────────────────────────────────────────────────────
def run_test(question: str) -> dict:
    print(f"\n{'═' * 62}")
    print(f"  Q: {question}")
    print("═" * 62)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    messages = result.get("messages", [])
    tool_calls_made = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made += 1
                sql = tc.get("args", {}).get("sql", "")
                print(f"\n  [tool call #{tool_calls_made}] {tc['name']}")
                print(f"  SQL: {sql}")

        if isinstance(msg, ToolMessage):
            print(f"  Result:\n{msg.content}")

    # Final answer is the last AIMessage with content
    final = next(
        (m for m in reversed(messages)
         if isinstance(m, AIMessage) and m.content),
        None,
    )
    answer = final.content.strip() if final else "(no answer)"
    print(f"\n  [answer]\n{answer}")

    return {"success": bool(final), "tool_calls": tool_calls_made}


results = []
for q in QUESTIONS:
    r = run_test(q)
    results.append(r)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═' * 62}")
print(f"  SUMMARY  ({MODEL} via Deep Agents)")
print("═" * 62)
passed   = sum(1 for r in results if r["success"])
avg_calls = sum(r["tool_calls"] for r in results) / len(results)
print(f"  Passed            : {passed}/{len(QUESTIONS)}")
print(f"  Avg tool calls/Q  : {avg_calls:.1f}")
for i, (q, r) in enumerate(zip(QUESTIONS, results), 1):
    status = "OK" if r["success"] else "FAIL"
    print(f"  [{status}] Q{i} — {r['tool_calls']} call(s)  |  {q[:50]}")
print()

con.close()
