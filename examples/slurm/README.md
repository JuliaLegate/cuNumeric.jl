# Distributed cuNumeric on SLURM

Launch one Legate rank per SLURM task with
[`SlurmClusterManager`](https://github.com/kleinhenz/SlurmClusterManager.jl),
run distributed cuNumeric across them, and shut down cleanly.

## Files
- `run.sh` — one dual-mode script. Run as `sbatch run.sh` it's the **driver**;
  SlurmClusterManager also re-invokes it (with `--worker`) as each **worker's**
  launcher, where it binds Julia to the chosen interconnect. Edit the `#SBATCH`
  directives at the top.
- `run_slurm.jl` — starts the ranks, runs a per-worker check, calls
  `cuNumeric.finalize_workers`.

## Requirements
- `julia` on `PATH`, with the depot and this project reachable on **every** node
  (a shared/NFS filesystem), so all ranks load the same cuNumeric build.
- Submit from this directory: `sbatch run.sh`.

## Multi-node networking
Julia `Distributed` (control plane) and Legate/UCX (data plane) both open
connections between nodes on arbitrary ports. Two common cluster gotchas:

1. **Multi-homed nodes / firewalled default interface.** Julia and UCX may pick
  an interface that peers can't reach. Pin everything to the right interconnect
  by setting `CUNUMERIC_NET_SUBNET` to the **leading octets shared by that
  network's IP addresses** (if your nodes are `10.77.0.4`, `10.77.0.5`, …, use
  `10.77.`):
  ```bash
  export CUNUMERIC_NET_SUBNET=10.77.
  sbatch run.sh
  ```
  Each node finds its own NIC whose IPv4 starts with that prefix, `--bind-to`s
  it, and sets `UCX_NET_DEVICES` to it. A prefix (not an interface name) is used
  because the interface name for the same network can differ per node. Leave
  unset on single-network clusters.

2. **Firewalled ephemeral ports.** If cross-node connections hang even with the
  right interface, the compute-node firewall may be dropping arbitrary TCP
  ports; ask your admins to allow inter-node traffic on the interconnect.

## Scaling / customizing
- Change `--nodes` / `--ntasks-per-node` in `run.sh` for more ranks.
- Replace the `@everywhere` body in `run_slurm.jl` with your own distributed
  cuNumeric computation.
