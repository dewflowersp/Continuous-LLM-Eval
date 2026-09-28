#!/usr/bin/env bash
#
# Launch the monitoring dashboard.
#
# Working directory is this folder so relative data paths and Streamlit's
# script-dir import root stay consistent.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x .venv/bin/streamlit ]; then
    printf '\033[1;31merror: .venv is missing. Run ./setup.sh first.\033[0m\n' >&2
    exit 1
fi

exec .venv/bin/streamlit run app.py \
    --server.headless true \
    --server.port "${DEMO_PORT:-8502}" \
    --browser.gatherUsageStats false \
    --theme.base light
