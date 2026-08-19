#!/usr/bin/env bash
#SBATCH --job-name=cunumeric-distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2      # Legate ranks per node
#SBATCH --cpus-per-task=2        # >1 so Legate has room for utility threads
#SBATCH --time=00:30:00
#
# Distributed cuNumeric over SLURM via SlurmClusterManager. One dual-role script:
#   `sbatch run.sh`    -> DRIVER: runs run_slurm.jl
#   as worker exename  -> SLURM re-invokes it with `--worker`; binds that
#                         worker's Julia to the right NIC (run_slurm.jl points here)
#
# Needs `julia` on PATH and the depot + project on shared/NFS. Submit from here.
#
# Firewalled cluster? Pin distributed traffic to one interconnect.
# Set CUNUMERIC_NET_SUBNET to the leading octets shared by that network's IPs;
# each node then uses its NIC whose IPv4 starts with it. Unset otherwise.
#     export CUNUMERIC_NET_SUBNET=10.77.     # e.g. NICs 10.77.0.4, 10.77.0.5, ...

# If a subnet is pinned, bind this node to its interface on that subnet.
# `ip -4 -o addr` prints one address per line, e.g.:
#     "3: eno8403    inet 10.77.0.4/24 brd 10.77.0.255 scope global eno8403"
bind_args=()
if [[ -n "${CUNUMERIC_NET_SUBNET:-}" ]]; then
    net_line=$(ip -4 -o addr | grep -F " inet ${CUNUMERIC_NET_SUBNET}" | head -n1)
    read -r _idx iface _inet cidr _rest <<<"$net_line"   # iface=eno8403  cidr=10.77.0.4/24
    ip=${cidr%/*}                                        # ip=10.77.0.4  (drop /24)

    if [[ -n "$ip" ]]; then
        bind_args=(--bind-to "$ip")
        export UCX_NET_DEVICES="$iface"     # pin Legate's UCX transport too
    fi
fi

if [[ " $* " == *" --worker "* ]]; then
    # WORKER role: SlurmClusterManager exename (args end in `--worker`)
    exec "${CUNUMERIC_JULIA:-julia}" "${bind_args[@]}" "$@"
fi

# DRIVER role: sbatch entry point (runs in the submit dir)
# The driver coordinates workers; it must not start a Legate runtime.
export LEGATE_SKIP_RUNTIME=true
exec julia "${bind_args[@]}" --project=. run_slurm.jl
