# SPMD cuNumeric under MPI

Every rank is a Legate rank, wired by Realm's MPI bootstrap — **no driver process**. All
ranks run `run_mpi.jl`; `cuNumeric.rand`/`sum`/… act on one distributed array, so a global
reduction is identical on every rank. `cuNumeric.addprocs` is disabled here (the Distributed
path is separate; see `../distributed`). Use `cuNumeric.is_main_process()` to guard IO and
`cuNumeric.mpi_rank()` / `mpi_size()` for rank info.

## SLURM
```bash
sbatch run.sh
```
`run.sh` launches ranks with `srun`. SLURM sets `SLURM_NTASKS`, which Legate reads to bring
up its multi-node network — nothing else needed.

## Local (no SLURM)
```bash
mpirun -np 4 ./mpi_launcher.sh env LEGATE_CONFIG="--cpus 2" julia --project=. run_mpi.jl
```
The wrapper is only for **MPICH/Hydra `mpirun`**: Legate reads the launcher's world-size env
when its library loads (before any Julia runs), and MPICH exposes only `PMI_*`, which Legate
ignores — so `mpi_launcher.sh` maps `PMI_* → OMPI_COMM_WORLD_*` first. OpenMPI `mpirun` and
SLURM `srun` set those natively, so there the wrapper is a no-op and you can run `julia`
directly.

## vs `../distributed`
| | `distributed/` | `mpi/` (here) |
|---|---|---|
| Processes | 1 driver + N workers | N ranks, no driver |
| Legate networking | Realm **p2p** | Realm **MPI** |
| `addprocs` | required | disabled |
