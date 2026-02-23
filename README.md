# LLM Agent Framework Benchmark

Benchmarking overhead of popular Python agent frameworks for local LLM tool calling
(Qwen3 8B via Ollama, RTX 4070).

## Results

| Framework | Avg time / question | Avg tool calls | Overhead vs raw |
|---|---|---|---|
| **Raw Ollama** (Python client) | **3.5s** | 1.0 | 1x (baseline) |
| **Deep Agents** (LangGraph) | 19.1s | 1.0 | ~5.5x |
| **Google ADK** (LiteLLM) | 31.7s | 1.2 | ~9x |

All 3 frameworks answered all 5 questions correctly. The difference is **pure framework overhead**.

## Setup

- **Model:** Qwen3 8B (`qwen3:8b`) via [Ollama](https://ollama.com)
- **GPU:** NVIDIA RTX 4070 (8 GB VRAM)
- **Task:** Write and execute DuckDB SQL queries to answer 5 business questions about a sales table
- **Metric:** Wall-clock time per question (model inference + framework overhead)

## Why the gap?

- **Raw Ollama** talks directly to the model with zero translation layers
- **Deep Agents / LangGraph** adds ~5.5x overhead — worth it when you need durable execution, memory, or subagents
- **Google ADK** routes through LiteLLM -> OpenAI-compat -> Ollama (3 layers), making it ~9x slower for local inference

## Questions tested

1. What is the total revenue per category?
2. Which region had the highest total revenue?
3. Show monthly revenue for each month.
4. What are the top 3 best-selling products by quantity?
5. Which product has the best revenue-to-quantity ratio?

## File structure

```
bench/
├── utils.py                # shared: DB setup, formatter, logging helpers
├── tool_calling_test.py    # benchmark: Raw Ollama Python client
├── deepagents_test.py      # benchmark: Deep Agents + LangChain Ollama
├── adk_test.py             # benchmark: Google ADK + LiteLLM
├── run_all.sh              # run all three benchmarks in sequence
├── bench_results.log       # auto-generated results log (git-ignored)
└── .gitignore
```

`utils.py` contains shared constants (`MODEL`, `QUESTIONS`, `SEP`),
the in-memory DuckDB setup, table formatter, and `append_log()` so each
script stays focused on its own framework.

## Requirements

- [Ollama](https://ollama.com) running locally
- [uv](https://docs.astral.sh/uv/) — each script declares its own dependencies
  via PEP 723 inline metadata, so `uv run` installs them automatically

Pull the model once:

```bash
ollama pull qwen3:8b
```

## Run

### All benchmarks at once

```bash
./run_all.sh
```

`run_all.sh` will:
1. Verify Ollama is reachable
2. Pull `qwen3:8b` if not already present
3. Run all three scripts with `uv run`
4. Append one summary line per run to `bench_results.log`

### Individual scripts

```bash
uv run tool_calling_test.py
uv run deepagents_test.py
uv run adk_test.py
```

`uv` creates an isolated virtual environment and installs declared
dependencies automatically — no manual `pip install` needed.

## bench_results.log

Each run appends a one-line summary, e.g.:

```
2026-02-22 09:00:00 | raw_ollama                 | 5/5 passed | avg  3.50s | 1.0 calls/q
2026-02-22 09:01:00 | deep_agents                | 5/5 passed | avg 19.10s | 1.0 calls/q
2026-02-22 09:02:00 | google_adk                 | 5/5 passed | avg 31.70s | 1.2 calls/q
```

The log file is git-ignored so it stays local.
