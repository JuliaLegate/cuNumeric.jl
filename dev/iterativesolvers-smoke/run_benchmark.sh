#!/usr/bin/env bash
# Usage: BENCH_PROJECT=/path/to/env bash run_benchmark.sh 8192 16384 ...
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${BENCH_PROJECT:?Set BENCH_PROJECT to the environment created by setup_benchmark.jl}"
JULIA=${JULIA:-julia}
BENCH_BACKENDS=${BENCH_BACKENDS:-"cuNumeric CuArray DaggerPatched"}
export BENCH_ELTYPE=${BENCH_ELTYPE:-Float32}
export BENCH_LAYOUT=row
export BENCH_PROFILE=false
export LEGATE_AUTO_CONFIG=${LEGATE_AUTO_CONFIG:-0}
export LEGATE_CONFIG=${LEGATE_CONFIG:-"--gpus 1 --cpus 2 --fbmem 22000 --sysmem 65536 --zcmem 1024"}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
if (( $# == 0 )); then set -- 8192 16384 32768; fi
for n in "$@"; do
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > 1 )) || { echo "Invalid N: $n" >&2; exit 2; }
done
output=${BENCH_OUTPUT:-"$PWD/cg-results-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv="$output/results.csv"
[[ ! -e "$csv" ]] || { echo "Refusing to overwrite $csv" >&2; exit 2; }
printf '%s\n' 'backend,eltype,n,iterations,median_ms,min_ms,max_ms,relative_residual,samples_ms,workspace_config,profiled' > "$csv"
{
    git -C "$script_dir" rev-parse HEAD
    "$JULIA" --version
    nvidia-smi
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nLEGATE_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$LEGATE_CONFIG"
    "$JULIA" --startup-file=no --project="$BENCH_PROJECT" -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)'
} > "$output/environment.txt" 2>&1
failed=0
for backend in $BENCH_BACKENDS; do
    for n in "$@"; do
        log="$output/$BENCH_ELTYPE-$backend-$n.log"
        if "$JULIA" -t4 --startup-file=no --project="$BENCH_PROJECT" \
            "$script_dir/benchmark_cg.jl" "$backend" "$n" 2>&1 | tee "$log"; then
            awk -v workspace="${CUBLAS_WORKSPACE_CONFIG:-<default>}" \
                '/^RESULT,/ {sub(/^RESULT,/, ""); print $0 "," workspace ",false"}' "$log" >> "$csv"
        else
            echo "Failed: $backend N=$n (see $log)" >&2
            failed=1
        fi
    done
done
echo "Results: $csv"
exit "$failed"
