#!/bin/bash
# Wrapper script to run Julia with conda mode, avoiding library conflicts
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX is not set. Please activate a conda environment."
    exit 1
fi

export CONDA_ENV="${CONDA_PREFIX}"

# Preload conda libraries to avoid conflicts with JLL artifacts
# This prevents HDF5_jll symbol errors
export LD_PRELOAD="${CONDA_ENV}/lib/libstdc++.so:${CONDA_ENV}/lib/libhdf5.so:${CONDA_ENV}/lib/liblegate.so"

echo $PWD
julia --project=$PWD -e "using Pkg; Pkg.add(\"LegatePreferences\"); using LegatePreferences; LegatePreferences.use_conda(\"${CONDA_ENV}\");"
julia --project=$PWD -e "using Pkg; Pkg.add(\"CNPreferences\"); using CNPreferences; CNPreferences.use_conda(\"${CONDA_ENV}\");"

# Run Julia with any provided arguments
exec julia --project=$PWD "$@"