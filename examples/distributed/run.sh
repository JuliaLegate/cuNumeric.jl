#!/usr/bin/env bash
#SBATCH --job-name=cunumeric-distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2      # Legate ranks per node
#SBATCH --cpus-per-task=2        # room for Legate utility threads
#SBATCH --time=00:30:00
#
# Dual-role: `sbatch run.sh` is the driver; SlurmClusterManager re-invokes it `--worker`
# as each worker's launcher. Needs julia + project on shared/NFS. Submit from here.
# Optional: CUNUMERIC_NET_SUBNET pins traffic to one interconnect (see README).

# Bind this node to its NIC on the pinned subnet, if set.
bind_args=()
if [[ -n "${CUNUMERIC_NET_SUBNET:-}" ]]; then
    net_line=$(ip -4 -o addr | grep -F " inet ${CUNUMERIC_NET_SUBNET}" | head -n1)
    read -r _idx iface _inet cidr _rest <<<"$net_line"   # e.g. iface=eno8403 cidr=10.77.0.4/24
    ip=${cidr%/*}
    if [[ -n "$ip" ]]; then
        bind_args=(--bind-to "$ip")
        export UCX_NET_DEVICES="$iface"     # pin Legate's UCX transport too
    fi
fi

# Worker role: SlurmClusterManager exename (args end in `--worker`).
if [[ " $* " == *" --worker "* ]]; then
    exec "${CUNUMERIC_JULIA:-julia}" "${bind_args[@]}" "$@"
fi

# Driver role: coordinates workers, must not start a Legate runtime.
export LEGATE_SKIP_RUNTIME=true
exec julia "${bind_args[@]}" --project=. run_slurm.jl
