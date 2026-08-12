#!/usr/bin/env bash

set -euo pipefail

readonly JLL_PIPELINE=".buildkite/jll.pipeline.yml"
readonly DEVELOPER_PIPELINE=".buildkite/developer.pipeline.yml"
readonly WRAPPER_PATH="lib/cunumeric_jl_wrapper"
readonly WRAPPER_BASE_BRANCH="main"

branch="${BUILDKITE_BRANCH:-}"
base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}"
pull_request="${BUILDKITE_PULL_REQUEST:-false}"
message="${BUILDKITE_MESSAGE:-}"

run_jll=true
run_developer=true

# Keep both suites for main and PRs into main. For non-main PRs, select the
# suite whose wrapper matches the code under test.
if [[ "$branch" != "main" && "$base_branch" != "main" ]]; then
    if [[ "$message" =~ \[skip[[:space:]]jll\] ]]; then
        echo "Skipping JLL GPU CI because the build message contains [skip jll]."
        run_jll=false
    elif [[ "$pull_request" != "false" && -n "$base_branch" ]]; then
        base_ref="refs/remotes/origin/$WRAPPER_BASE_BRANCH"
        # The published wrapper JLL tracks main, so compare against main even
        # when the pull request targets develop.
        git fetch --no-tags origin "+refs/heads/${WRAPPER_BASE_BRANCH}:${base_ref}"

        if git diff --quiet "${base_ref}...HEAD" -- "$WRAPPER_PATH"; then
            echo "No wrapper changes detected against origin/$WRAPPER_BASE_BRANCH; using JLL GPU CI."
            run_developer=false
        else
            diff_status=$?
            if ((diff_status == 1)); then
                echo "Wrapper changes detected against origin/$WRAPPER_BASE_BRANCH; using developer GPU CI."
                run_jll=false
            else
                echo "Could not determine whether the wrapper changed against origin/$WRAPPER_BASE_BRANCH." >&2
                exit "$diff_status"
            fi
        fi
    fi
fi

# Each dynamic upload is inserted immediately after this job, so upload the
# developer group first to keep the JLL group first when both suites run.
if [[ "$run_developer" == "true" ]]; then
    buildkite-agent pipeline upload "$DEVELOPER_PIPELINE"
fi

if [[ "$run_jll" == "true" ]]; then
    buildkite-agent pipeline upload "$JLL_PIPELINE"
fi
