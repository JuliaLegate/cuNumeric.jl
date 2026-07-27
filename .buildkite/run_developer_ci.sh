#!/usr/bin/env bash

set -euo pipefail

CI_PROJECT="$(mktemp -d)"

case "${CUNUMERIC_FUSION:-}" in
    on | off) ;;
    *)
        echo "CUNUMERIC_FUSION must be 'on' or 'off'" >&2
        exit 2
        ;;
esac

CMAKE_VERSION="3.30.7"
CMAKE_ROOT="$(mktemp -d)"
CMAKE_INSTALLER="$CMAKE_ROOT/cmake-installer.sh"
curl --fail --silent --show-error --location \
    --output "$CMAKE_INSTALLER" \
    "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh"
sh "$CMAKE_INSTALLER" --skip-license --prefix="$CMAKE_ROOT"
export PATH="$CMAKE_ROOT/bin:$PATH"
cmake --version

LEGATE_BRANCH_INPUT="${BUILDKITE_MESSAGE:-}"
if [[ "${BUILDKITE_PULL_REQUEST:-false}" =~ ^[0-9]+$ ]]; then
    echo "Reading Legate branch override from pull request #$BUILDKITE_PULL_REQUEST"
    PR_BODY="$(
        curl --fail --silent --show-error --location \
            --header "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/JuliaLegate/cuNumeric.jl/pulls/$BUILDKITE_PULL_REQUEST" |
            python3 -c 'import json, sys; print(json.load(sys.stdin).get("body") or "")'
    )"
    LEGATE_BRANCH_INPUT+=$'\n'"$PR_BODY"
fi

shopt -s nocasematch
LEGATE_BRANCH=""
if [[ "$LEGATE_BRANCH_INPUT" =~ legate[-_]branch:[[:space:]]*([A-Za-z0-9._/-]+) ]]; then
    LEGATE_BRANCH="${BASH_REMATCH[1]}"
fi
shopt -u nocasematch

if [[ -n "$LEGATE_BRANCH" ]]; then
    LEGATE_CHECKOUT="$(mktemp -d)/Legate.jl"
    echo "Using Legate.jl branch override: $LEGATE_BRANCH"
    git clone --depth 1 --branch "$LEGATE_BRANCH" \
        https://github.com/JuliaLegate/Legate.jl.git "$LEGATE_CHECKOUT"
    git -C "$LEGATE_CHECKOUT" log -1 --format="Legate checkout: %D (%H)"
    (
        cd "$LEGATE_CHECKOUT"
        julia --color=yes --project="$CI_PROJECT" -e '
            using Pkg
            Pkg.develop(PackageSpec(path = "lib/LegatePreferences"))
            Pkg.develop(PackageSpec(path = "."))
            using LegatePreferences
            LegatePreferences.use_developer_mode()
            Pkg.build("Legate")
        '
    )
else
    echo "No Legate.jl branch override - using JLL"
fi

julia --color=yes --project="$CI_PROJECT" -e '
    using Pkg
    Pkg.develop(PackageSpec(path = "lib/CNPreferences"))
    using CNPreferences
    CNPreferences.use_developer_mode()
    CNPreferences.set_broadcast_fusion!(ENV["CUNUMERIC_FUSION"] == "on")
    CNPreferences.set_broadcast_fusion_min_ops!(1)
    Pkg.develop(PackageSpec(path = "."))
    Pkg.build("cuNumeric")
'

julia --color=yes --project=test -e '
    using Pkg
    Pkg.develop(PackageSpec(path = "lib/CNPreferences"))
    using CNPreferences
    CNPreferences.use_developer_mode()
    CNPreferences.set_broadcast_fusion!(ENV["CUNUMERIC_FUSION"] == "on")
    CNPreferences.set_broadcast_fusion_min_ops!(1)
'

julia --color=yes --project="$CI_PROJECT" -e '
    using Pkg
    Pkg.test("cuNumeric"; test_args = ["--quickfail"])
'
