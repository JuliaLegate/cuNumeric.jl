#!/usr/bin/env bash
#SBATCH --job-name=cunumeric-mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2      # Legate ranks per node
#SBATCH --cpus-per-task=2        # room for Legate utility threads
#SBATCH --time=00:30:00
#
# SPMD cuNumeric over SLURM: srun runs one Legate rank per task; Realm's MPI bootstrap wires
# them via SLURM's PMI. Legate sees SLURM_NTASKS, so no PMI->OMPI wrapper is needed here
# (that's only for local MPICH mpirun; see mpi_launcher.sh). Submit from here.
set -euo pipefail

export LEGATE_CONFIG="--cpus ${SLURM_CPUS_PER_TASK:-1}"
exec srun julia --project=. --threads="${SLURM_CPUS_PER_TASK:-1}" run_mpi.jl
