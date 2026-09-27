#!/usr/bin/env bash
# Exit 0 if the wrapper at HEAD matches RELEASED_COMMIT (the source of the
# released wrapper JLL), 1 if it differs.

set -euo pipefail

readonly WRAPPER_PATH="lib/cunumeric_jl_wrapper"
readonly RELEASED_COMMIT_FILE="$WRAPPER_PATH/RELEASED_COMMIT"

released="$(tr -d '[:space:]' < "$RELEASED_COMMIT_FILE")"
if ! git cat-file -e "${released}^{commit}" 2>/dev/null; then
    git fetch --no-tags --depth=1 origin "$released"
fi

git diff --quiet "$released" HEAD -- "$WRAPPER_PATH" ":(exclude)$RELEASED_COMMIT_FILE"
