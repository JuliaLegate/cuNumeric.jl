# Integration test: launch the SPMD payload under mpirun and assert the ranks connect.
# Skipped where mpirun is unavailable (e.g. CI without an MPI). Runs in fresh processes so
# it never touches the test runner's own Legate runtime / Distributed workers.
using Test

const NRANKS = 2

mpirun = Sys.which("mpirun")
worker = joinpath(@__DIR__, "mpi_worker.jl.inc")

if mpirun === nothing
    @info "mpirun not found; skipping MPI SPMD integration test"
    @test_skip false
else
    cmd = `$mpirun -np $NRANKS $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $worker`
    # Force the runtime on for the ranks even if the parent set LEGATE_SKIP_RUNTIME.
    p = run(
        ignorestatus(addenv(cmd,
            "LEGATE_CONFIG" => "--cpus 2", "LEGATE_SKIP_RUNTIME" => "false")),
    )
    @test success(p)
end
