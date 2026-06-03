#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <filename.jl> [--gpus <num_gpus>] [--cpus <num_cpus>] [extra_args...]"
    exit 1
fi

# Parse arguments
FILENAME=$1
shift

GPUS=0
CPUS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS=$2
            shift 2
            ;;
        --cpus)
            CPUS=$2
            shift 2
            ;;
        *)
            # Collect all other arguments as extra arguments
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate the filename exists
if [[ ! -f $FILENAME ]]; then
    echo "Error: File $FILENAME does not exist."
    exit 1
fi

# Inform user of the configuration
if [[ $GPUS -lt 0 ]]; then
    echo "GPUs invalid, using gpus = 0"
    exit
fi

if [[ $CPUS -lt 0 ]]; then
    echo "CPUs invalid, using cpus = 1"
    exit
fi

export LEGATE_AUTO_CONFIG=1
export LEGATE_CONFIG="--cpus=$CPUS --gpus=$GPUS"
export LEGATE_SHOW_CONFIG=1

export LD_LIBRARY_PATH=""

echo "Running $FILENAME with $CPUS CPUs and $GPUS GPUs"

CMD="julia --project='..' $FILENAME $GPUS ${EXTRA_ARGS[@]}"

printf "Running: %s\n" "$CMD"
eval "$CMD"
