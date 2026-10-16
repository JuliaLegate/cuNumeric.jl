# SPMD Gray-Scott under MPI (numerics from ../gray-scott.jl). Launch: see README. GS_N/GS_STEPS.
using cuNumeric

cuNumeric.Experimental(true)
include(joinpath(@__DIR__, "..", "gray-scott.jl"))

N = parse(Int, get(ENV, "GS_N", "512"))
n_steps = parse(Int, get(ENV, "GS_STEPS", "1000"))
cuNumeric.is_main_process() &&
    println("Gray-Scott SPMD: N=$N steps=$n_steps ranks=$(cuNumeric.mpi_size())")

gray_scott_run(N, min(n_steps, 5))   # warm up (compile)
cuNumeric.Legate.runtime_sync()
elapsed = @elapsed begin
    u = gray_scott_run(N, n_steps)
    cuNumeric.Legate.runtime_sync()
end
checksum = cuNumeric.allowscalar() do
    Float64(cuNumeric.sum(u)[])
end
cuNumeric.is_main_process() && println("done: $(round(elapsed; digits=3))s, sum(u)=$checksum")
