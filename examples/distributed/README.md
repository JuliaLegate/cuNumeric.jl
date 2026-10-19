# Distributed cuNumeric

A driver (Distributed proc 1, no Legate runtime) starts N Legate **worker** ranks over the
p2p bootstrap and drives them with `@everywhere`.

## Local — no scheduler
```bash
julia --project=. run_local.jl                       # 2 workers
CUNUMERIC_NWORKERS=4 julia --project=. run_local.jl
```
`addprocs(n)` starts workers on this machine. For multiple nodes without SLURM, use a host
list / `SSHManager`.

## SLURM
```bash
sbatch run.sh
```
`run.sh` is dual-role — the sbatch entry is the driver, and `SlurmClusterManager` re-invokes
it `--worker` per worker. `run_slurm.jl` runs one rank per task and calls `finalize_workers`.
Requires `julia` + this project on shared/NFS so every node loads the same build.

**Multi-homed / firewalled clusters:** set `CUNUMERIC_NET_SUBNET` to the leading octets of
the right network so each node binds its matching NIC (a prefix, since names differ per node):
```bash
export CUNUMERIC_NET_SUBNET=10.77.    # NICs 10.77.0.4, 10.77.0.5, ...
sbatch run.sh
```
