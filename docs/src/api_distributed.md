# Distributed Execution

cuNumeric can run across multiple processes and nodes using either an MPI-style SPMD
program or a Julia `Distributed` driver with Legate workers.

!!! warning "Experimental Feature"
    Distributed execution is experimental. Enable it in your program:
    ```julia
    cuNumeric.Experimental(true)
    ```

!!! tip "Prefer MPI"
    Use the MPI approach for new workloads. It maps directly onto `srun`/`mpirun`, has
    no separate driver process, and requires less lifecycle and network setup.

| | MPI (recommended) | `Distributed` |
|---|---|---|
| Process model | Every process is a Legate rank | One driver starts Legate workers |
| Bootstrap | Realm MPI | Realm peer-to-peer |
| Julia model | SPMD: every rank runs the program | Driver dispatches work with `@everywhere` |
| Best suited for | New multi-node jobs | Existing `Distributed.jl` orchestration |

## MPI (recommended)

Every rank runs the same program and participates in the same distributed computation.
Use `is_main_process()` to guard output and `mpi_rank()`/`mpi_size()` for rank information.

```julia
using cuNumeric
cuNumeric.Experimental(true)

a = cuNumeric.rand(1_000)
total = cuNumeric.sum(a)

cuNumeric.is_main_process() &&
    println("sum=$total across $(cuNumeric.mpi_size()) ranks")
```

A minimal Slurm launcher is:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2

export LEGATE_CONFIG="--cpus ${SLURM_CPUS_PER_TASK}"
srun julia --project=. --threads="${SLURM_CPUS_PER_TASK}" run_mpi.jl
```

Locally, launch with `mpirun -np 4 julia --project=. run_mpi.jl`. MPICH/Hydra users
should use the environment adapter in `examples/mpi/mpi_launcher.sh`. See
[`examples/mpi`](https://github.com/JuliaLegate/cuNumeric.jl/tree/main/examples/mpi) for
complete local and Slurm examples.

## Julia `Distributed`

This approach keeps process 1 as a driver without a Legate runtime. It starts configured
Legate workers and drives them using the standard `Distributed.jl` API.

```julia
using Distributed
using cuNumeric
cuNumeric.Experimental(true)

pids = cuNumeric.addprocs(2)
@everywhere pids println("worker=$(myid()) host=$(gethostname())")
cuNumeric.finalize_workers(pids)
```

Always call `finalize_workers` before the driver exits. On Slurm, pass a
`SlurmClusterManager.SlurmManager()` to `cuNumeric.addprocs`; the driver must set
`LEGATE_SKIP_RUNTIME=true`. See
[`examples/distributed`](https://github.com/JuliaLegate/cuNumeric.jl/tree/main/examples/distributed)
for the complete dual-role `sbatch` launcher and multi-homed network configuration.
