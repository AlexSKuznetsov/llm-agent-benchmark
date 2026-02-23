# LLM Agent Framework Benchmark

Benchmarking overhead of popular Python agent frameworks for local LLM tool calling (Qwen3 8B via Ollama, RTX 4070).

## Results

| Framework | Avg time / question | Avg tool calls |
|---|---|---|
| **Raw Ollama** (Python client) | **3.5s** | 1.0 |
| **Deep Agents** (LangGraph) | 19.1s | 1.0 |
| **Google ADK** (LiteLLM) | 31.7s | 1.2 |

All 3 frameworks answered all 5 questions correctly. The difference is **pure framework overhead**.

## Setup

- **Model:** Qwen3 8B (`qwen3:8b`) via [Ollama](https://ollama.com)
- **GPU:** NVIDIA RTX 4070 (8 GB VRAM)
- **Task:** Write and execute DuckDB SQL queries to answer 5 business questions about a sales table
- **Metric:** Wall-clock time per question (model inference + framework overhead)

## Why the gap?

- **Raw Ollama** talks directly to the model with zero translation layers
- **Deep Agents / LangGraph** adds ~5.5x overhead — worth it when you need durable execution, memory, or subagents
- **Google ADK** routes through LiteLLM to OpenAI-compat to Ollama (3 layers), making it ~9x slower for local inference

## Questions tested

1. What is the total revenue per category?
2. Which region had the highest total revenue?
3. Show monthly revenue for each month.
4. What are the top 3 best-selling products by quantity?
5. Which product has the best revenue-to-quantity ratio?

## Files

| File | Framework |
|---|---|
| tool_calling_test.py | Raw Ollama Python client |
| deepagents_test.py | Deep Agents + LangChain Ollama |
| adk_test.py | Google ADK + LiteLLM |

## Requirements

Install Ollama and pull the model:

    ollama pull qwen3:8b

Install Python deps:

    pip install ollama duckdb
    pip install deepagents langchain-ollama duckdb
    pip install google-adk[extensions] duckdb

## Run

    python tool_calling_test.py
    python deepagents_test.py
    python adk_test.py
