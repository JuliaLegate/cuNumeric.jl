#!/bin/bash
# Clone (or update) JuliaLegate/benchmarking into <repo>/benchmark.
#   ./scripts/get_benchmarks.sh [git ref]
# The benchmarks expect to live inside the cuNumeric.jl checkout, so keep the default location.

set -euo pipefail

REPO_URL=${CUNUMERIC_BENCH_URL:-https://github.com/JuliaLegate/benchmarking.git}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/benchmark"
REF=${1:-}

if [ -e "$DEST/.git" ]; then
    git -C "$DEST" fetch --tags origin
    if [ -z "$REF" ]; then
        git -C "$DEST" remote set-head origin --auto >/dev/null
        git -C "$DEST" checkout "$(git -C "$DEST" rev-parse --abbrev-ref origin/HEAD | sed 's|^origin/||')"
        git -C "$DEST" pull --ff-only
    fi
else
    git clone "$REPO_URL" "$DEST"
fi

if [ -n "$REF" ]; then
    git -C "$DEST" checkout "$REF"
fi

echo "Benchmarks at $DEST ($(git -C "$DEST" describe --tags --always))"
echo "Run: cd benchmark && ./run_smoke.sh   or   ./run_all.sh"
