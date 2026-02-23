#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen3:8b"
OLLAMA_URL="http://localhost:11434"

echo "=== LLM Agent Benchmark ==="
echo ""

# 1. Check Ollama is running
echo -n "Checking Ollama... "
if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; then
    echo "NOT running"
    echo "Start Ollama first:  ollama serve"
    exit 1
fi
echo "OK"

# 2. Check / pull model
echo -n "Checking model ${MODEL}... "
if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    echo "already present"
else
    echo "not found - pulling..."
    ollama pull "${MODEL}"
fi

# 3. Pre-install all dependencies (warms up uv venv cache)
echo ""
echo "--- Pre-installing dependencies ---"
for script in tool_calling_test.py deepagents_test.py adk_test.py; do
    echo -n "  ${script} ... "
    BENCH_WARMUP=1 uv run "${script}"
done

# 4. Run benchmarks
echo ""
echo "--- Raw Ollama ---"
uv run tool_calling_test.py

echo ""
echo "--- Deep Agents ---"
uv run deepagents_test.py

echo ""
echo "--- Google ADK ---"
uv run adk_test.py

echo ""
echo "=== Done! Results saved to bench_results.log ==="
