import json
import duckdb
import ollama

MODEL = "qwen3:8b"

# ── Sample data ──────────────────────────────────────────────────────────────
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

# ── Tool definition ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_duckdb_query",
            "description": (
                "Execute a SQL query against a DuckDB database and return results. "
                "The database has one table: sales(date DATE, product VARCHAR, "
                "category VARCHAR, quantity INTEGER, price DECIMAL, region VARCHAR)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Valid DuckDB SQL query to execute.",
                    }
                },
                "required": ["sql"],
            },
        },
    }
]


def run_duckdb_query(sql: str) -> str:
    try:
        result = con.execute(sql)
        cols = [desc[0] for desc in result.description]
        rows = result.fetchall()
        col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(cols)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        lines = [fmt.format(*cols), "  ".join("-" * w for w in col_widths)]
        lines += [fmt.format(*[str(v) for v in row]) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"


TOOL_MAP = {"run_duckdb_query": run_duckdb_query}

# ── Questions ─────────────────────────────────────────────────────────────────
QUESTIONS = [
    "What is the total revenue per category?",
    "Which region had the highest total revenue?",
    "Show monthly revenue for each month.",
    "What are the top 3 best-selling products by quantity?",
    "Which product has the best revenue-to-quantity ratio?",
]


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_test(question: str, max_iterations: int = 6) -> dict:
    print(f"\n{'═' * 62}")
    print(f"  Q: {question}")
    print("═" * 62)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst. Use the run_duckdb_query tool to answer "
                "questions. Always query the database — do not guess the numbers."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_calls_made = 0
    for iteration in range(max_iterations):
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            think=False,  # disable chain-of-thought for qwen3
        )
        msg = response.message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments
                sql  = args.get("sql", "")

                print(f"\n  [tool call #{tool_calls_made + 1}] {name}")
                print(f"  SQL: {sql}")

                result = TOOL_MAP[name](sql)
                print(f"  Result:\n{result}")

                messages.append({"role": "tool", "content": result})
                tool_calls_made += 1
        else:
            print(f"\n  [answer]\n{msg.content.strip()}")
            return {"success": True, "tool_calls": tool_calls_made}

    print(f"\n  [!] Reached max iterations ({max_iterations}) without final answer.")
    return {"success": False, "tool_calls": tool_calls_made}


# ── Run all tests ─────────────────────────────────────────────────────────────
results = []
for q in QUESTIONS:
    r = run_test(q)
    results.append(r)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═' * 62}")
print(f"  SUMMARY  ({MODEL})")
print("═" * 62)
passed = sum(1 for r in results if r["success"])
avg_calls = sum(r["tool_calls"] for r in results) / len(results)
print(f"  Passed   : {passed}/{len(QUESTIONS)}")
print(f"  Avg tool calls per question: {avg_calls:.1f}")
for i, (q, r) in enumerate(zip(QUESTIONS, results), 1):
    status = "OK" if r["success"] else "FAIL"
    print(f"  [{status}] Q{i} — {r['tool_calls']} tool call(s)  |  {q[:50]}")
print()

con.close()
