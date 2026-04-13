#!/bin/bash
# Run quick correctness test for all 24 NNLS solvers from the terminal.
#
# Usage:
#   ./test.sh

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v matlab &>/dev/null; then
    echo "Error: matlab not found on PATH" >&2
    exit 1
fi

matlab -batch "run('test.m')"
