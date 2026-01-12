using Distributed
addprocs(4)

@everywhere begin
    include("port.jl")
    setup_legate_env()

    if myid() != 1
        ENV["LEGATE_CONFIG"] = "--gpus=1 --fbmem=1000 --show-config --logging legate=debug,level=2  --show-progress  --show-memory-usage  --profile"
        using cuNumeric

        a = cuNumeric.rand(100)
        b = cuNumeric.rand(100)
        c = -(a + b)
    end
end
