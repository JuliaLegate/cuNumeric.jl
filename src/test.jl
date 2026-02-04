ENV["LEGATE_CONFIG"] = "--cpus=2 --show-config --logging legate=debug,level=2  --show-progress  --show-memory-usage  --profile"

using Distributed;
addprocs(4)

@everywhere begin
    if myid() != 1
        include("port.jl")
        setup_legate_env()

        using cuNumeric
        println("Number of runtimes: ", cuNumeric.get_number_of_runtimes())

        a = cuNumeric.rand(100)
        b = cuNumeric.rand(100)
        c = -(a + b)
    end
end
