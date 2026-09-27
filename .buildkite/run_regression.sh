#!/usr/bin/env bash
# Opt-in GPU performance comparison of this commit against a base branch:
# [regression-ci <branch>], else REGRESSION_BASE_BRANCH, else the branch the PR
# targets, else main. Both sides run this commit's harness and regression.toml.

set -euo pipefail

requested="$(buildkite-agent meta-data get regression-base-branch --default "" 2>/dev/null || true)"
pr_base="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}"
readonly BASE_BRANCH="${requested:-${REGRESSION_BASE_BRANCH:-${pr_base:-main}}}"
readonly THRESHOLD="${REGRESSION_THRESHOLD:-10}"
# Bounds each base harness pass so a hang on the base cannot use up the step.
readonly BASE_TIMEOUT="${REGRESSION_BASE_TIMEOUT:-3600}"

candidate="$PWD"
harness="$candidate/benchmark"
env_dir="$harness/environments/cunumeric"
out="$candidate/regression"
base="$(mktemp -d)/base"

rm -rf "$out"
mkdir -p "$out"

git fetch --no-tags origin "+refs/heads/$BASE_BRANCH:refs/remotes/origin/$BASE_BRANCH"
git worktree add --detach "$base" "origin/$BASE_BRANCH"
trap 'git -C "$candidate" worktree remove --force "$base"' EXIT
git submodule update --init benchmark

julia --color=yes --project="$harness" -e 'using Pkg; Pkg.instantiate()'
mkdir -p "$harness/results"

# Point the harness's cuNumeric environment at one checkout. Wrapper overrides
# and precompiled wrapper bindings from the other side must not leak across.
bind_checkout() {
    local source=$1 mode=$2
    local depot
    depot="$(julia --startup-file=no -e 'print(DEPOT_PATH[1])')"
    rm -rf "$depot"/packages/*/*/override \
           "$depot"/compiled/v*/{cuNumeric,Legate,cunumeric_jl_wrapper_jll,legate_jl_wrapper_jll}
    rm -f "$env_dir/Manifest.toml" "$env_dir/LocalPreferences.toml"
    julia --color=yes --project="$env_dir" -e '
        using Pkg
        Pkg.develop([PackageSpec(path = ARGS[1]), PackageSpec(path = ARGS[2])])
        Pkg.instantiate()
    ' "$source" "$source/lib/CNPreferences"
    if [[ "$mode" == developer ]]; then
        julia --color=yes --project="$env_dir" -e '
            using CNPreferences, Pkg
            CNPreferences.use_developer_mode()
            Pkg.build("cuNumeric")
        '
    fi
}

result_dirs() { find "$harness/results" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'; }

run_side() {
    local side=$1 source=$2 mode=$3
    local limit=()
    [[ "$side" == base ]] && limit=(timeout --signal=KILL "$BASE_TIMEOUT")
    echo "--- :julia: $side ($mode wrapper)"
    bind_checkout "$source" "$mode"
    for fusion in on off; do
        local before new
        before="$(result_dirs)"
        (cd "$harness" && "${limit[@]}" julia --color=yes --project=. run.jl \
            --config="$candidate/.buildkite/regression.toml" --fusion="$fusion") ||
            echo "Harness reported failures ($side, fusion $fusion)."
        new="$(comm -13 <(sort <<<"$before") <(result_dirs | sort) | head -1)"
        if [[ -n "$new" ]]; then
            mkdir -p "$out/$side"
            mv "$harness/results/$new" "$out/$side/fusion-$fusion"
        fi
    done
}

# Build a side's wrapper from source when it differs from the release its own
# checkout records. A base without RELEASED_COMMIT is checked against ours.
wrapper_mode() {
    local dir=$1
    [[ -f "$dir/scripts/wrapper_changed.sh" ]] || dir="$candidate"
    if (cd "$dir" && scripts/wrapper_changed.sh "$2" >&2); then
        echo jll
    else
        local status=$?
        ((status == 1)) || exit "$status"
        echo developer
    fi
}

base_mode="$(wrapper_mode "$base" "$(git -C "$base" rev-parse HEAD)")" || base_mode=developer
candidate_mode="$(wrapper_mode "$candidate" HEAD)"
if [[ "$base_mode" == developer || "$candidate_mode" == developer ]]; then
    source .buildkite/install_cmake.sh
fi

echo "Comparing against $BASE_BRANCH."
# Base failures only leave its results uncompared; the candidate must pass.
run_side base "$base" "$base_mode" || echo "Base side failed; its results are not compared."
run_side candidate "$candidate" "$candidate_mode"

echo "--- :bar_chart: Compare"
status=0
julia --startup-file=no "$candidate/.buildkite/compare_regression.jl" \
    "$out/base" "$out/candidate" --threshold="$THRESHOLD" --base="$BASE_BRANCH" \
    --out="$out/report.md" ||
    status=$?

if [[ -f "$out/report.md" ]] && command -v buildkite-agent >/dev/null; then
    style=$([[ $status == 0 ]] && echo success || echo error)
    buildkite-agent annotate --context regression --style "$style" < "$out/report.md"
fi
exit "$status"
