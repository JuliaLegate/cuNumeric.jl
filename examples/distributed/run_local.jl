# Local distributed cuNumeric -- no scheduler. Starts N worker ranks on this machine
# (LocalManager), wires them via p2p, runs a per-worker check, shuts down.
#   julia --project=. run_local.jl
#   CUNUMERIC_NWORKERS=4 julia --project=. run_local.jl
using Distributed
using cuNumeric

cuNumeric.Experimental(true)

n = parse(Int, get(ENV, "CUNUMERIC_NWORKERS", "2"))
pids = cuNumeric.addprocs(n)  # n worker processes on this host

@everywhere pids begin  # cuNumeric already initialized on each worker
    println("worker=$(myid()) host=$(gethostname())")
end

cuNumeric.finalize_workers(pids)  # collective shutdown before the driver exits
