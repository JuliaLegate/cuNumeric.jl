#!/usr/bin/env bash
#SBATCH --job-name=cunumeric
#SBATCH --nodes=2                    # Run on exactly 2 nodes
#SBATCH --ntasks-per-node=4          # Run 4 processes per node (8 total tasks)
#SBATCH --cpus-per-task=1            # Assign 1 CPU core per process
#SBATCH --mem-per-cpu=2G             # Request 2GB RAM per CPU core
#SBATCH --nodelist=dubliner,roquefort
#SBATCH --time=00:30:00


# The batch driver coordinates workers but should not consume a GPU runtime.
export LEGATE_SKIP_RUNTIME=true

exec julia --project=. run_slurm.jl
