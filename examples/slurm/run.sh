#!/usr/bin/env bash
#SBATCH --job-name=cunumeric
#SBATCH --nodes=2                    # Run on exactly 2 nodes
#SBATCH --ntasks-per-node=2          # Run 2 worker processes per node (4 total)
#SBATCH --cpus-per-task=2            # 2 CPU cores per worker (Legate needs >1)
#SBATCH --mem-per-cpu=2G             # Request 2GB RAM per CPU core
#SBATCH --nodelist=jarlsberg,limburger  # CPU-only nodes (no /pool) -> NFS install
#SBATCH --time=00:30:00

# Load the tank (NFS) install so the batch driver and every srun-launched worker
# use the same Julia + depot. Also clears the system-CUDA LD_LIBRARY_PATH.
source /tank/david/cunumeric-slurm/env.sh

# The batch driver coordinates workers but should not start a Legate runtime.
export LEGATE_SKIP_RUNTIME=true

cd "$CUNUMERIC_SLURM_ROOT/cuNumeric.jl/examples/slurm"
exec julia --project=. run_slurm.jl
