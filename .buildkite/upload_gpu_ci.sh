#!/usr/bin/env bash

set -euo pipefail

readonly JLL_PIPELINE=".buildkite/jll.pipeline.yml"
readonly DEVELOPER_PIPELINE=".buildkite/developer.pipeline.yml"

branch="${BUILDKITE_BRANCH:-}"
base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}"
pull_request="${BUILDKITE_PULL_REQUEST:-false}"
message="${BUILDKITE_MESSAGE:-}"

run_jll=true
run_developer=true

if [[ "$message" =~ \[skip[[:space:]]ci\] ]]; then
    echo "Skipping all GPU CI because the build message requests it."
    exit 0
fi

if [[ "$message" =~ \[skip[[:space:]]jll\] ]]; then
    echo "Skipping JLL GPU CI because the build message contains [skip jll]."
    run_jll=false
fi
if [[ "$message" =~ \[skip[[:space:]]dev\] ]]; then
    echo "Skipping developer GPU CI because the build message contains [skip dev]."
    run_developer=false
fi

# Keep both suites for main and PRs into main. Otherwise use the JLL suite only
# when the wrapper matches the released JLL source.
if [[ "$branch" != "main" && "$base_branch" != "main" ]]; then
    if scripts/wrapper_changed.sh; then
        echo "Wrapper matches the released JLL source; using JLL GPU CI."
        run_developer=false
    else
        diff_status=$?
        if ((diff_status == 1)); then
            echo "Wrapper differs from the released JLL source; using developer GPU CI."
            run_jll=false
        else
            echo "Could not determine whether the wrapper matches the released JLL source." >&2
            exit "$diff_status"
        fi
    fi
fi

# Opt-in performance comparison: [regression-ci] in the commit message or the
# pull request title/body compares against the PR's base branch, and
# [regression-ci <branch>] against <branch>.
opt_in="$message"
if [[ "$pull_request" =~ ^[0-9]+$ ]]; then
    opt_in+=$'\n'"$(
        curl --fail --silent --show-error --location \
            --header "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/JuliaLegate/cuNumeric.jl/pulls/$pull_request" |
            python3 -c 'import json, sys; pr = json.load(sys.stdin); print(pr.get("title") or "", pr.get("body") or "")'
    )" || true
fi
if [[ "$opt_in" =~ \[regression-ci([[:space:]]+([A-Za-z0-9._/-]+))?\] ]]; then
    regression_base="${BASH_REMATCH[2]}"
    if [[ -n "$regression_base" ]]; then
        buildkite-agent meta-data set regression-base-branch "$regression_base"
    fi
    echo "Uploading benchmark regression CI (base: ${regression_base:-PR base branch})."
    buildkite-agent pipeline upload .buildkite/regression.pipeline.yml
fi

# Each dynamic upload is inserted immediately after this job, so upload the
# developer group first to keep the JLL group first when both suites run.
if [[ "$run_developer" == "true" ]]; then
    buildkite-agent pipeline upload "$DEVELOPER_PIPELINE"
fi

if [[ "$run_jll" == "true" ]]; then
    buildkite-agent pipeline upload "$JLL_PIPELINE"
fi
