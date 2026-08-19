# run_slurm.jl
using Distributed
using SlurmClusterManager
using cuNumeric

cuNumeric.Experimental(true)

cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))

pids = cuNumeric.addprocs(
    SlurmManager();
    exeflags=["--threads=$cpus"],
    env=[
        "LEGATE_SKIP_RUNTIME" => "false",
        "LEGATE_CONFIG" => "--cpus $cpus",
    ],
)

@everywhere pids begin
    # cuNumeric is already initialized here.
    println("worker=$(myid()) host=$(gethostname()) cpus=$(get(ENV, "SLURM_CPUS_PER_TASK", "?"))")
end
