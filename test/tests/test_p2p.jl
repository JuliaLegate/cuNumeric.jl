ENV["LEGATE_CONFIG"] = "--cpus=2 --show-config --logging legate=debug,level=2 --show-progress  --show-memory-usage  --profile"

using Distributed
using cuNumeric

addprocs(4)

# Set up peer-specific env vars and load cuNumeric on workers
cuNumeric.init_workers()

# Test on workers - cuNumeric will load with p2p networking
@everywhere workers() begin
    println("Worker $(myid()): Testing cuNumeric")
    println("Number of runtimes: ", cuNumeric.get_number_of_runtimes())
    println("P2P plugin: ", get(ENV, "REALM_UCP_BOOTSTRAP_PLUGIN", "not set"))
    println("Self info: ", get(ENV, "WORKER_SELF_INFO", "not set"))

    a = cuNumeric.rand(100)
    b = cuNumeric.rand(100)
    c = -(a + b)

    println("Worker $(myid()): Test complete, result shape: ", size(c))
end
