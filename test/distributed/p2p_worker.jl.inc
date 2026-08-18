ENV["LEGATE_CONFIG"] = "--cpus=2 --show-config --logging legate=debug,level=2 --show-progress --show-memory-usage --profile"

using Distributed
using cuNumeric

cuNumeric.Experimental(false)
try
    cuNumeric.addprocs(1)
    error("cuNumeric.addprocs should require Experimental mode")
catch err
    err isa ArgumentError || rethrow()
end

cuNumeric.Experimental(true)
nworkers = parse(Int, get(ENV, "CUNUMERIC_DISTRIBUTED_TEST_WORKERS", "4"))
cuNumeric.addprocs(nworkers)

@everywhere workers() begin
    println("Worker $(myid()): Testing cuNumeric")
    println("Number of runtimes: ", cuNumeric.get_number_of_runtimes())
    println("P2P plugin: ", get(ENV, "REALM_UCP_BOOTSTRAP_PLUGIN", "not set"))
    println("Self info: ", get(ENV, "WORKER_SELF_INFO", "not set"))

    a = cuNumeric.ones(Float64, 100)
    b = cuNumeric.ones(Float64, 100)
    c = -(a + b)
    checksum = cuNumeric.sum(c)[]
    @assert checksum == -200.0 "unexpected distributed result: $checksum"

    println("Worker $(myid()): Test complete, checksum: ", checksum)
end
