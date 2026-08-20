#!/usr/bin/env bash

set -euo pipefail

SOURCE_CHECKOUT="$(pwd -P)"
CI_WORKSPACE="$(mktemp -d "${TMPDIR:-/tmp}/cunumeric-developer-ci.XXXXXX")"

cleanup() {
    local status=$?
    local diagnostic suffix="${BUILDKITE_JOB_ID:-$$}"
    trap - EXIT
    set +e
    cd "$CI_WORKSPACE"
    for diagnostic in deps/*.log deps/*.err *.log; do
        [[ -f "$diagnostic" ]] || continue
        cp "$diagnostic" \
            "$SOURCE_CHECKOUT/${diagnostic%.*}.${suffix}.${diagnostic##*.}"
    done
    cd "$SOURCE_CHECKOUT"
    [[ "$CI_WORKSPACE" == "${TMPDIR:-/tmp}"/cunumeric-developer-ci.* ]] && \
        rm -rf -- "$CI_WORKSPACE"
    exit "$status"
}
trap cleanup EXIT

git -C "$SOURCE_CHECKOUT" archive "${BUILDKITE_COMMIT:-HEAD}" | tar -x -C "$CI_WORKSPACE"
mkdir "$CI_WORKSPACE/.tmp"
export TMPDIR="$CI_WORKSPACE/.tmp"
cd "$CI_WORKSPACE"

# Migrate caches whose generated CMake target still points at an old checkout.
DEPOT="$(julia --startup-file=no -e 'print(DEPOT_PATH[1])')"
JLCXX_DEV="$DEPOT/dev/libcxxwrap_julia_jll"
JLCXX_SOURCE="$DEPOT/dev/libcxxwrap-julia"
if [[ -d "$JLCXX_DEV" ]] && \
   { [[ ! -f "$JLCXX_SOURCE/include/jlcxx/jlcxx.hpp" ]] || \
     [[ ! -f "$JLCXX_DEV/override/lib/libcxxwrap_julia.so" ]] || \
     [[ ! -f "$JLCXX_DEV/override/lib/libcxxwrap_julia_stl.so" ]]; }; then
    rm -rf -- "$JLCXX_DEV"
fi

.buildkite/run_developer_ci.sh
