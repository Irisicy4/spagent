#!/usr/bin/env bash
# Collect results from spagent/test subdirs and generate LaTeX summary table.
set -e
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
TEST_BASE=$(dirname "$SCRIPT_DIR")/test

echo "Scanning test dir: $TEST_BASE"
python3 collect_results.py
python3 generate_latex_table.py
echo "Done. Output: $SCRIPT_DIR/results_table.tex and results_summary.json"
