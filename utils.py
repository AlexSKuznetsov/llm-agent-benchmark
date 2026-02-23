"""Shared constants, DB setup, and helpers for all benchmark scripts."""

import duckdb
from datetime import datetime
from pathlib import Path

MODEL    = "qwen3:8b"
LOG_FILE = Path(__file__).parent / "bench_results.log"

QUESTIONS = [
    "What is the total revenue per category?",
    "Which region had the highest total revenue?",
    "Show monthly revenue for each month.",
    "What are the top 3 best-selling products by quantity?",
    "Which product has the best revenue-to-quantity ratio?",
]

SEP = chr(9552) * 62


def setup_db() -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection pre-loaded with sample sales data."""
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
    return con


def fmt_table(result) -> str:
    """Format a DuckDB query result as a plain-text table string."""
    cols = [d[0] for d in result.description]
    rows = result.fetchall()
    if not rows:
        return "No results."
    widths = [
        max(len(str(c)), max(len(str(r[i])) for r in rows))
        for i, c in enumerate(cols)
    ]
    fmt   = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*cols), "  ".join("-" * w for w in widths)]
    lines += [fmt.format(*[str(v) for v in row]) for row in rows]
    return "\n".join(lines)


def make_query_runner(con: duckdb.DuckDBPyConnection):
    """Return a run_duckdb_query function bound to the given connection."""
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
        try:
            return fmt_table(con.execute(sql))
        except Exception as e:
            return f"Query error: {e}"
    return run_duckdb_query


def print_summary(name: str, results: list, times: list) -> None:
    passed    = sum(1 for r in results if r["success"])
    avg_calls = sum(r["tool_calls"] for r in results) / len(results)
    avg_time  = sum(times) / len(times)
    print()
    print(SEP)
    print(f"  SUMMARY  ({name})")
    print(SEP)
    print(f"  Passed           : {passed}/{len(results)}")
    print(f"  Avg tool calls/Q : {avg_calls:.1f}")
    print(f"  Avg time/Q       : {avg_time:.2f}s")
    for i, (q, r) in enumerate(zip(QUESTIONS, results), 1):
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] Q{i} - {r['tool_calls']} call(s)  {r['time']:.1f}s  |  {q[:45]}")
    print()


def append_log(name: str, results: list, times: list) -> None:
    """Append a one-line benchmark summary to bench_results.log."""
    passed    = sum(1 for r in results if r["success"])
    avg_calls = sum(r["tool_calls"] for r in results) / len(results)
    avg_time  = sum(times) / len(times)
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts} | {name:<26} | {passed}/{len(results)} passed"
        f" | avg {avg_time:5.2f}s | {avg_calls:.1f} calls/q\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(f"  appended to {LOG_FILE.name}")
