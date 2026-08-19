# Distributed cuNumeric launcher (run under SLURM via run.sh).
#
# Starts one Legate rank per SLURM task with SlurmClusterManager, runs a trivial
# per-worker check, then shuts the ranks down collectively. This is a template:
# the only cluster-specific knob is CUNUMERIC_NET_SUBNET (see run.sh / README).
using Distributed
using SlurmClusterManager
using cuNumeric

cuNumeric.Experimental(true)

cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
subnet = get(ENV, "CUNUMERIC_NET_SUBNET", "")   # e.g. "10.77."; empty = cluster default

worker_env = Pair{String,String}[
    "LEGATE_SKIP_RUNTIME" => "false",
    "LEGATE_CONFIG" => "--cpus $cpus",
    # Let each worker find the same Julia the driver is running.
    "CUNUMERIC_JULIA" => joinpath(Sys.BINDIR, Base.julia_exename()),
]
# Pass the interconnect subnet through; run.sh's "--worker" role resolves each
# node's interface from it and sets --bind-to + UCX_NET_DEVICES per node.
isempty(subnet) || push!(worker_env, "CUNUMERIC_NET_SUBNET" => subnet)

pids = cuNumeric.addprocs(
    SlurmManager();
    # Workers are launched by run.sh in its "--worker" role (it binds each worker
    # to the CUNUMERIC_NET_SUBNET interface when set). Same file as the driver.
    exename=joinpath(@__DIR__, "run.sh"),
    exeflags=["--project=$(Base.active_project())", "--threads=$cpus"],
    env=worker_env,
)

@everywhere pids begin
    # cuNumeric is already initialized on each worker.
    println("worker=$(myid()) host=$(gethostname()) cpus=$(get(ENV, "SLURM_CPUS_PER_TASK", "?"))")
end

# Coordinated collective shutdown across all ranks before the driver exits.
cuNumeric.finalize_workers(pids)
