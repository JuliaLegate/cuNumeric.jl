# Distributed cuNumeric over SLURM (run via run.sh). Starts one Legate rank per task,
# runs a per-worker check, shuts down. Only cluster knob: CUNUMERIC_NET_SUBNET (see README).
using Distributed
using SlurmClusterManager
using cuNumeric

cuNumeric.Experimental(true)

cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
subnet = get(ENV, "CUNUMERIC_NET_SUBNET", "")   # e.g. "10.77."; empty = cluster default

worker_env = Pair{String,String}[
    "LEGATE_SKIP_RUNTIME" => "false",
    "LEGATE_CONFIG" => "--cpus $cpus",
    "CUNUMERIC_JULIA" => joinpath(Sys.BINDIR, Base.julia_exename()),  # same julia as driver
]
isempty(subnet) || push!(worker_env, "CUNUMERIC_NET_SUBNET" => subnet)

pids = cuNumeric.addprocs(
    SlurmManager();
    exename=joinpath(@__DIR__, "run.sh"),  # run.sh's --worker role binds each worker's NIC
    exeflags=["--project=$(Base.active_project())", "--threads=$cpus"],
    env=worker_env,
)

@everywhere pids begin  # cuNumeric already initialized on each worker
    println("worker=$(myid()) host=$(gethostname()) cpus=$(get(ENV, "SLURM_CPUS_PER_TASK", "?"))")
end

cuNumeric.finalize_workers(pids)  # collective shutdown before the driver exits
