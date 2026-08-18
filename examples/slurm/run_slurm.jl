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
        "LEGATE_CONFIG" => "--gpus 1 --cpus $cpus",
    ],
)

@everywhere pids begin
    # cuNumeric is already initialized here.
    println("worker=$(myid()) host=$(gethostname()) gpu=$(get(ENV, "CUDA_VISIBLE_DEVICES", ""))")
end
