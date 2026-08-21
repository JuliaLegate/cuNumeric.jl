using Test

if get(ENV, "CUNUMERIC_PARALLEL_TEST_RUNNER", "0") == "1"
    project = Base.active_project()
    script = joinpath(@__DIR__, "p2p_worker.jl.inc")
    run(`$(Base.julia_cmd()) --startup-file=no --project=$project $script`)
    @test true
else
    include(joinpath(@__DIR__, "p2p_worker.jl.inc"))
end
