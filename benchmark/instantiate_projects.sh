#!/usr/bin/env bash
set -euo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
julia_bin="${CUNUMERIC_BENCH_JULIA:-julia}"

cd "$benchmark_dir"

for environment in cuda jacc dagger; do
    echo "Instantiating environments/$environment"
    "$julia_bin" --project="environments/$environment" \
        -e 'using Pkg; Pkg.instantiate()'
done

echo "Developing local packages and instantiating environments/cunumeric"
"$julia_bin" --project="environments/cunumeric" \
    -e 'using Pkg; Pkg.develop(path=ARGS[1]); Pkg.develop(path=ARGS[2]); Pkg.instantiate()' \
    ../ ../lib/CNPreferences
