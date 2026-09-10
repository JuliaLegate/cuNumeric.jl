#!/usr/bin/env bash
set -euo pipefail

# Optional command after -- is an external cluster launcher, passed as argv.
# All resource/memory/network options remain configurable via LEGATE_CONFIG.
mode=${1:?Usage: run.sh cpu|one-gpu|multi-gpu|multi-node [-- launcher arguments...]}
shift
case "$mode" in
    cpu) gpus=0; expected=0 ;;
    one-gpu) gpus=1; expected=1 ;;
    multi-gpu) gpus=${CUNUMERIC_LINALG_GPUS_PER_NODE:-4}; expected=$gpus ;;
    multi-node)
        gpus=${CUNUMERIC_LINALG_GPUS_PER_NODE:-4}
        expected=${CUNUMERIC_LINALG_EXPECT_GPUS:?Set the total GPU count across nodes}
        ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
if [[ ! $gpus =~ ^[0-9]+$ || ! $expected =~ ^[0-9]+$ ]]; then
    echo "GPU counts must be nonnegative integers" >&2
    exit 2
fi
if [[ $mode == multi-gpu && $gpus -lt 2 ]]; then
    echo "multi-gpu requires at least two GPUs" >&2
    exit 2
fi
if [[ $mode == multi-node && ( $gpus -lt 1 || $expected -le $gpus ) ]]; then
    echo "multi-node expects GPUs on more than one node" >&2
    exit 2
fi
launcher=()
if [[ ${1:-} == -- ]]; then
    shift
    launcher=("$@")
elif [[ $# -gt 0 ]]; then
    echo "Launcher arguments must follow --" >&2
    exit 2
fi
if [[ $mode == multi-node && ${#launcher[@]} -eq 0 ]]; then
    echo "multi-node needs an external launcher after --" >&2
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
logdir=${CUNUMERIC_LINALG_LOGDIR:-"$repo/linalg-results/$mode-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$logdir"
logdir=$(cd "$logdir" && pwd)
export LEGATE_AUTO_CONFIG=0
export LEGATE_SKIP_RUNTIME=false
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="${LEGATE_CONFIG:---gpus $gpus --cpus 2 --profile}"
export CUNUMERIC_LINALG_EXPECT_GPUS=$expected
julia=${CUNUMERIC_LINALG_JULIA:-julia}
seconds=${CUNUMERIC_LINALG_TIMEOUT:-1800}
extra=()
[[ ${CUNUMERIC_LINALG_PRODUCTION:-0} == 1 ]] && extra+=(--production)

printf 'Mode: %s\nLEGATE_CONFIG: %s\nLogs: %s\n' "$mode" "$LEGATE_CONFIG" "$logdir"
mkdir -p "$logdir/acceptance"
export CUNUMERIC_LINALG_VERBOSE=1
LEGATE_CONFIG="$LEGATE_CONFIG --logdir $logdir/acceptance" \
timeout --kill-after=30s "${seconds}s" "${launcher[@]}" "$julia" --project="$repo" \
    "$repo/scripts/linalg/acceptance.jl" "${extra[@]}" 2>&1 | tee "$logdir/acceptance.log"

# Separate launches also isolate communicators and ensure failures cannot hang
# the acceptance driver forever. Nonzero exits/timeouts remain test failures.
backends=(single tiled)
[[ $expected -gt 1 ]] && backends+=(mp)
for backend in "${backends[@]}"; do
    for op in solve cholesky; do
        [[ $backend == tiled && $op == solve ]] && continue
        logfile="$logdir/failure-$backend-$op.log"
        mkdir -p "$logdir/failure-$backend-$op"
        LEGATE_CONFIG="$LEGATE_CONFIG --logdir $logdir/failure-$backend-$op" \
        timeout --kill-after=30s "${seconds}s" "${launcher[@]}" "$julia" --project="$repo" \
            "$repo/scripts/linalg/failure.jl" "$op" "$backend" 2>&1 | tee "$logfile"
        grep -q 'EXPECTED_LINALG_FAILURE:' "$logfile"
    done
done
echo "Numerical and failure checks passed. Review the task profiles before accepting the topology."
