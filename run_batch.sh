#!/usr/bin/env bash
set -euo pipefail
source /workspace/venv/bin/activate
export PROMPT_API_BASE="${PROMPT_API_BASE:-https://manus-api-server.onrender.com}"
python /workspace/ai-batch/automation.py
