#!/bin/bash
# Copyright 2025 Northwestern University,
#                   Carnegie Mellon University University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author(s): David Krasowska <krasow@u.northwestern.edu>
#            Ethan Meitz <emeitz@andrew.cmu.edu>

set -e

# Check if exactly one argument is provided
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

CUNUMERIC_ROOT_DIR=$1  # First argument

# Check if the provided argument is a valid directory
if [[ ! -d "$CUNUMERIC_ROOT_DIR" ]]; then
    echo "Error: '$CUNUMERIC_ROOT_DIR' is not a valid directory."
    exit 1
fi

JULIA=${JULIA:-julia}
JULIA_PATH=$(command -v "$JULIA")

if [ -z "$JULIA_PATH" ]; then
  echo "Error: $JULIA is not installed or not in PATH."
  exit 1
fi

echo "Using $JULIA at: $JULIA_PATH"

GIT_REPO="https://github.com/JuliaInterop/libcxxwrap-julia.git"
COMMIT_HASH="ee8a49b403ced7669c8fa56cec860f567f6510aa" #(v14.11)
JULIA_CXXWRAP_SRC=$CUNUMERIC_ROOT_DIR/lib/libcxxwrap-julia

if [ ! -d "$JULIA_CXXWRAP_SRC" ]; then
    mkdir -p "$CUNUMERIC_ROOT_DIR/lib"
    git clone "$GIT_REPO" "$JULIA_CXXWRAP_SRC"
fi

cd "$JULIA_CXXWRAP_SRC"
git fetch --tags
git checkout $COMMIT_HASH

# find julia dependency path
JULIA_DEP_PATH=$("$JULIA_PATH" --startup-file=no -e 'print(DEPOT_PATH[1])')

# https://github.com/JuliaInterop/libcxxwrap-julia/tree/v0.13.3?tab=readme-ov-file#configuring-and-building
JULIA_CXXWRAP_DEV=$JULIA_DEP_PATH/dev/libcxxwrap_julia_jll
JULIA_CXXWRAP=$JULIA_CXXWRAP_DEV/override

# Keep the resolved environment and developed JLL checkout. Only its generated
# override is disposable. Avoid importing/precompiling a possibly broken JLL
# before its replacement libraries have been built.
cd "$CUNUMERIC_ROOT_DIR"
JULIA_PKG_PRECOMPILE_AUTO=0 "$JULIA_PATH" --startup-file=no --project="$CUNUMERIC_ROOT_DIR" -e '
    using Pkg
    checkout = joinpath(DEPOT_PATH[1], "dev", "libcxxwrap_julia_jll")
    if isdir(checkout)
        Pkg.develop(path=checkout)
    else
        Pkg.develop(PackageSpec(name="libcxxwrap_julia_jll"); shared=true)
    end
'

rm -rf "$JULIA_CXXWRAP"
mkdir -p "$JULIA_CXXWRAP"

cmake -S "$JULIA_CXXWRAP_SRC" -B "$JULIA_CXXWRAP" \
    -DJulia_EXECUTABLE="$JULIA_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$JULIA_CXXWRAP" --parallel 16
