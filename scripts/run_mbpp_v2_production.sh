#!/bin/bash
# Production launcher for the fresh MBPP benchmark.
# exec replaces the shell with python, so signals (and pkill -f matching this
# script name) reach the runner itself — no orphaned children.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/mbpp_v2_pilot.py \
    --schema mbpp-v2-production/1 \
    --models gpt-5.4 ds-v4-flash glm-5.3-flash qwen38-flash gpt-6-luna \
    --out data/mbpp_v2/production \
    --transport-retries 2 \
    --workers 16
