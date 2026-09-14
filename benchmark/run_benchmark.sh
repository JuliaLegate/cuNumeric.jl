#!/bin/bash

set -euo pipefail

MODEL=""
GPUS=""
CPUS=""
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model=*)
            MODEL=${1#*=}
            shift
            ;;
        --gpus=*)
            GPUS=${1#*=}
            shift
            ;;
        --cpus=*)
            CPUS=${1#*=}
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: unknown runner option '$1'." >&2
            exit 2
            ;;
    esac
done

if [[ -z $MODEL || -z $GPUS || -z $CPUS || $# -eq 0 ]]; then
    echo "Usage: $0 --model=<name> --gpus=<n> --cpus=<n> [--verbose] -- <command> [args...]" >&2
    exit 2
fi

if ! [[ $GPUS =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPUs must be a positive integer; got '$GPUS'." >&2
    exit 2
fi

if ! [[ $CPUS =~ ^[0-9]+$ ]]; then
    echo "Error: CPUs must be a nonnegative integer; got '$CPUS'." >&2
    exit 2
fi

# A worker process has exactly one execution-model identity. Reject nested or
# accidentally reused launch environments instead of allowing a worker to load
# a second model under a misleading result label.
if [[ -n ${CUNUMERIC_BENCH_ACTIVE_MODEL:-} && $CUNUMERIC_BENCH_ACTIVE_MODEL != "$MODEL" ]]; then
    echo "Error: refusing to launch model '$MODEL' inside model '$CUNUMERIC_BENCH_ACTIVE_MODEL'." >&2
    exit 2
fi
export CUNUMERIC_BENCH_ACTIVE_MODEL=$MODEL
export CUNUMERIC_BENCH_GPUS=$GPUS
export CUNUMERIC_BENCH_CPUS=$CPUS

if [[ $VERBOSE == 1 ]]; then
    printf 'Running [%s]:' "$MODEL"
    printf ' %q' "$@"
    printf '\n'
fi

exec "$@"
