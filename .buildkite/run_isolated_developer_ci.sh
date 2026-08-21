#!/usr/bin/env bash

set -euo pipefail

SOURCE_CHECKOUT="$(pwd -P)"
CI_TMP_ROOT="${TMPDIR:-/tmp}"
CI_WORKSPACE="$(mktemp -d "$CI_TMP_ROOT/cunumeric-developer-ci.XXXXXX")"

cleanup() {
    local status=$?
    local pattern
    trap - EXIT
    set +e
    cd "$CI_WORKSPACE"
    for pattern in "deps/*.log" "deps/*.err" "*.log"; do
        compgen -G "$pattern" >/dev/null && buildkite-agent artifact upload "$pattern"
    done
    cd "$SOURCE_CHECKOUT"
    [[ "$CI_WORKSPACE" == "$CI_TMP_ROOT"/cunumeric-developer-ci.* ]] && \
        rm -rf -- "$CI_WORKSPACE"
    exit "$status"
}
trap cleanup EXIT

git -C "$SOURCE_CHECKOUT" archive "${BUILDKITE_COMMIT:-HEAD}" | tar -x -C "$CI_WORKSPACE"
mkdir "$CI_WORKSPACE/.tmp"
export TMPDIR="$CI_WORKSPACE/.tmp"
cd "$CI_WORKSPACE"
.buildkite/run_developer_ci.sh
