# SPMD cuNumeric under MPI

Every rank is a Legate rank (Realm's MPI bootstrap) — **no driver**. All ranks run
`run_mpi.jl` over one distributed array, so a global reduction matches on every rank.
`addprocs` is disabled here. Use `is_main_process()` to guard IO, and `mpi_rank()` /
`mpi_size()` for rank info.

## SLURM
```bash
sbatch run.sh
```
`run.sh` launches ranks with `srun`; Legate reads `SLURM_NTASKS` and brings up its network.

## Local — no scheduler
```bash
mpirun -np 4 ./mpi_launcher.sh env LEGATE_CONFIG="--cpus 2" julia --project=. run_mpi.jl
```
`mpi_launcher.sh` is only needed for **MPICH/Hydra `mpirun`**, which exposes rank info as
`PMI_*`; it maps that to the `OMPI_COMM_WORLD_*` Legate reads at library load. OpenMPI and
SLURM `srun` set those natively — there it's a no-op and you can run `julia` directly.

## vs `../distributed`
| | `distributed/` | `mpi/` |
|---|---|---|
| Processes | 1 driver + N workers | N ranks, no driver |
| Networking | Realm **p2p** | Realm **MPI** |
| `addprocs` | required | disabled |
