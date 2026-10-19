# SPMD payload run on every rank under mpirun (see spmd.jl). Asserts the ranks form one
# Legate cluster and that addprocs is refused. Any failed @assert exits nonzero -> test fails.

# Map MPICH PMI_* -> the OMPI_* vars Legate reads, before loading cuNumeric.
# OpenMPI mpirun / SLURM srun already set these, so the guard skips (matches mpi_launcher.sh).
if haskey(ENV, "PMI_SIZE") && !haskey(ENV, "OMPI_COMM_WORLD_SIZE") &&
   !haskey(ENV, "MV2_COMM_WORLD_SIZE") && !haskey(ENV, "SLURM_NTASKS")
    ENV["OMPI_COMM_WORLD_SIZE"] = ENV["PMI_SIZE"]
    ENV["OMPI_COMM_WORLD_RANK"] = get(ENV, "PMI_RANK", "0")
end
get!(ENV, "LEGATE_CONFIG", "--cpus 2")

using Distributed
using cuNumeric

cuNumeric.Experimental(true)

sz = cuNumeric.mpi_size()
@assert cuNumeric.under_mpi() "under_mpi() should be true under mpirun"
@assert sz == 2 "expected 2 ranks, got $sz"
@assert 0 <= cuNumeric.mpi_rank() < sz "rank out of range"
@assert cuNumeric.is_main_process() == (cuNumeric.mpi_rank() == 0)

# All ranks joined one cluster (== independent single-rank runtimes would each report 1).
nrt = cuNumeric.get_number_of_runtimes()
@assert nrt == sz "ranks not connected: get_number_of_runtimes()=$nrt, size=$sz"

# One distributed array -> global reduction is identical on every rank.
s = cuNumeric.allowscalar() do
    cuNumeric.sum(cuNumeric.ones(Float64, 1000))[]
end
@assert s == 1000.0 "unexpected global sum $s"

# The Distributed addprocs path is disabled under MPI.
try
    cuNumeric.addprocs(2)
    error("addprocs should be refused under MPI")
catch err
    occursin("launched under MPI", sprint(showerror, err)) || rethrow()
end

cuNumeric.mpi_rank() == 0 && println("MPI SPMD test OK ($sz ranks)")
