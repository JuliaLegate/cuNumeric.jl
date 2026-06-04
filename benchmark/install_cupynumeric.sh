#!/bin/bash
# Install a cupynumeric conda env matching the cupynumeric_jll our project resolves.
# The conda package and the JLL share the calendar-versioning scheme (e.g. 25.10),
# so we pin major.minor (patch ignored) and install from the legate channel.
#
# Usage:
#   ./install_cupynumeric.sh                 # create a fresh env named cupynumeric-bench-<ver>
#   ./install_cupynumeric.sh --name myenv    # override the env name
#   ./install_cupynumeric.sh --into existing # install into an existing env instead of creating one
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME=""
INTO_ENV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            ENV_NAME=$2
            shift 2
            ;;
        --into)
            INTO_ENV=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--name <env>] [--into <existing-env>]"
            exit 1
            ;;
    esac
done

# Resolve the JLL version Julia actually instantiated for this project, then keep
# major.minor only — conda packages are not published per patch.
echo "Detecting cupynumeric_jll version from the benchmark project..."
VER=$(cd "$SCRIPT_DIR" && julia --project -e '
using Pkg
for (_, info) in Pkg.dependencies()
    info.name == "cupynumeric_jll" || continue
    v = info.version
    isnothing(v) && continue
    println("$(v.major).$(v.minor)")
end' | tail -1)

if [[ -z "$VER" ]]; then
    echo "Error: could not detect cupynumeric_jll version. Has the project been instantiated?"
    exit 1
fi

echo "cupynumeric_jll major.minor: $VER"
SPEC="cupynumeric=$VER.*"

if [[ -n "$INTO_ENV" ]]; then
    echo "Installing $SPEC into existing env '$INTO_ENV'..."
    conda install -y -n "$INTO_ENV" -c conda-forge -c legate "$SPEC"
    echo "Done. Activate with: conda activate $INTO_ENV"
    exit 0
fi

[[ -z "$ENV_NAME" ]] && ENV_NAME="cupynumeric-bench-$VER"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Env '$ENV_NAME' already exists with $SPEC; nothing to do."
    echo "Activate with: conda activate $ENV_NAME"
    exit 0
fi

echo "Creating env '$ENV_NAME' with $SPEC..."
conda create -y -n "$ENV_NAME" -c conda-forge -c legate "$SPEC"

echo "Done. Activate with: conda activate $ENV_NAME"
