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

shopt -s nocasematch
LEGATE_BRANCH=""
if [[ "${BUILDKITE_MESSAGE:-}" =~ legate[-_]branch:[[:space:]]*([A-Za-z0-9._/-]+) ]]; then
    LEGATE_BRANCH="${BASH_REMATCH[1]}"
fi
shopt -u nocasematch

if [[ -n "$LEGATE_BRANCH" ]]; then
    LEGATE_CHECKOUT="$(mktemp -d)/Legate.jl"
    echo "Using Legate.jl branch override: $LEGATE_BRANCH"
    git clone --depth 1 --branch "$LEGATE_BRANCH" \
        https://github.com/JuliaLegate/Legate.jl.git "$LEGATE_CHECKOUT"
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
    Pkg.develop(PackageSpec(path = "."))
    Pkg.build("cuNumeric")
'

julia --color=yes --project=test -e '
    using Pkg
    Pkg.develop(PackageSpec(path = "lib/CNPreferences"))
    using CNPreferences
    CNPreferences.use_developer_mode()
    CNPreferences.set_broadcast_fusion!(ENV["CUNUMERIC_FUSION"] == "on")
'

julia --color=yes --project="$CI_PROJECT" -e '
    using Pkg
    Pkg.test("cuNumeric"; test_args = ["--quickfail"])
'
