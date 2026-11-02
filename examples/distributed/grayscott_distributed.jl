# Gray-Scott over distributed p2p (run via run.sh). Driver starts the ranks; every worker runs
# ../gray-scott.jl's kernel collectively and reports a checksum. GS_N/GS_STEPS.
using Distributed
using SlurmClusterManager
using cuNumeric

cuNumeric.Experimental(true)

cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
subnet = get(ENV, "CUNUMERIC_NET_SUBNET", "")

worker_env = Pair{String,String}[
    "LEGATE_SKIP_RUNTIME" => "false",
    "LEGATE_CONFIG" => "--cpus $cpus",
    "CUNUMERIC_JULIA" => joinpath(Sys.BINDIR, Base.julia_exename()),
]
isempty(subnet) || push!(worker_env, "CUNUMERIC_NET_SUBNET" => subnet)

pids = cuNumeric.addprocs(
    SlurmManager();
    exename=joinpath(@__DIR__, "run.sh"),
    exeflags=["--project=$(Base.active_project())", "--threads=$cpus"],
    env=worker_env,
)

N = parse(Int, get(ENV, "GS_N", "128"))
n_steps = parse(Int, get(ENV, "GS_STEPS", "100"))
kernel = joinpath(@__DIR__, "..", "gray-scott.jl")

@everywhere pids include($kernel)
@everywhere pids begin
    global GS_SUM = cuNumeric.allowscalar() do
        Float64(cuNumeric.sum(gray_scott_run($N, $n_steps))[])
    end
end
gs = remotecall_fetch(() -> GS_SUM, first(pids))
println("distributed Gray-Scott: N=$N steps=$n_steps workers=$(length(pids)) sum(u)=$gs")

cuNumeric.finalize_workers(pids)
