#!/usr/bin/env bash
# Reproduce the project's exact Python environment (CLAUDE.md section 12, ADR-5).
#
# Usage: scripts/setup_env.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv python install 3.11
uv sync --all-groups --locked

echo
echo "Environment ready in .venv/ (Python $(uv run python --version))."
echo "Activate with: source .venv/bin/activate"
echo "Or run commands directly with: uv run <command>"
