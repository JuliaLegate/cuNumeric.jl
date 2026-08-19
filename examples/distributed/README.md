# Distributed cuNumeric

A driver (Distributed proc 1, no Legate runtime) starts N Legate **worker** ranks, wires
them with the p2p bootstrap, and drives them via `@everywhere`.

## Local (no scheduler)
```bash
julia --project=. run_local.jl                      # 2 workers
CUNUMERIC_NWORKERS=4 julia --project=. run_local.jl
```
`run_local.jl` starts workers on this machine with `cuNumeric.addprocs(n)`. For multiple
nodes without SLURM, swap `addprocs(n)` for a host list / `SSHManager`.

## SLURM
```bash
sbatch run.sh
```
`run.sh` is dual-role: the sbatch entry is the **driver**; `SlurmClusterManager` re-invokes
it `--worker` as each worker's launcher. `run_slurm.jl` starts one rank per task and calls
`finalize_workers`. Needs `julia` + this project on a shared/NFS filesystem so every node
loads the same build.

### Pinning the interconnect (multi-homed / firewalled clusters)
Julia and UCX may pick a NIC that peers can't reach. Set `CUNUMERIC_NET_SUBNET` to the
leading octets of the right network; each node binds its matching NIC and sets
`UCX_NET_DEVICES`. A prefix (not an interface name) is used because names differ per node.
```bash
export CUNUMERIC_NET_SUBNET=10.77.    # for NICs 10.77.0.4, 10.77.0.5, ...
sbatch run.sh
```
If connections still hang, the compute-node firewall may be dropping ephemeral TCP ports.
