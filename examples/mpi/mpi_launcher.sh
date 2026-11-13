#!/usr/bin/env bash
# MPICH/Hydra mpirun launcher: `mpirun -np N ./mpi_launcher.sh julia --project=. script.jl`.
# Legate reads the launcher's world-size env when its library loads (before any Julia runs),
# and MPICH exposes only PMI_*, which Legate ignores -- so map it to OMPI_COMM_WORLD_* here.
# OpenMPI mpirun and SLURM srun set those natively, making this a no-op.
set -euo pipefail

if [[ -n "${PMI_SIZE:-}" && -z "${OMPI_COMM_WORLD_SIZE:-}" && -z "${MV2_COMM_WORLD_SIZE:-}" && -z "${SLURM_NTASKS:-}" ]]; then
    export OMPI_COMM_WORLD_SIZE="$PMI_SIZE"
    export OMPI_COMM_WORLD_RANK="${PMI_RANK:-0}"
fi

exec "$@"
