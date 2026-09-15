include("../src/cunumeric/benchmarks/cg.jl")
@testset "CG periodic checks and harness integration" begin
    for B in (ConjugateGradientBenchmark,ConjugateGradientAccelerated), T in (Float32,Float64), every in (1,4,30)
        b=B{T}(;N=17,check_every=every,max_iter=60)
        s=only(initialize(b;mod=Base))
        @test run!(b,s) % every==0
        A=Tridiagonal(ones(T,16),fill(T(4),17),ones(T,16))
        @test s.x ≈ A\fill(T(0.5),17) rtol=(T==Float32 ? 3e-5 : 3e-8)
        @test norm(A*s.x .- T(0.5)) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(17)/2
        first=copy(s.x); run!(b,s)
        @test s.x==first
    end
    s=only(initialize(ConjugateGradientBenchmark{Float64}(;N=17);mod=Base))
    @test_throws ErrorException run!(ConjugateGradientBenchmark{Float64}(;N=17,max_iter=3),s)
    @test run!(ConjugateGradientBenchmark{Float64}(;N=17,max_iter=1),s)==1
    @test s.x ≈ fill(17/(12*17-4),17)
    @test run!(ConjugateGradientBenchmark{Float64}(;N=17,check_every=30,max_iter=17),s)==17
    config=joinpath(@__DIR__,"../benchmarks_cg.toml")
    gs,specs=parse_config(config)
    raw=TOML.parsefile(config)
    @test first(specs).kwargs==Dict(:check_every=>10,:max_iter=>1000)
    runs=plan_runs(specs,gs,raw,parse_plot_groups(config),10^12)
    @test Set(r.model for r in runs)==Set((:cunumeric,:jacc))
    @test Set((r.spec.name,r.model) for r in runs)==Set([
        ("cg",:cunumeric),("cg",:jacc),("cg_accelerated",:cunumeric)])
    @test length(unique((r.N,r.M) for r in runs))==1
    @test plan_manifest(runs,10^12,raw)["runs"][1]["kwargs"]==raw["cg"][1]["kwargs"]
    b=build_benchmark(ConjugateGradientBenchmark,Float64,17,1;check_every=4,max_iter=40)
    @test (b.check_every,b.max_iter)==(4,40)
    request=WorkerRequest(1,1,"cg","Float64",17,1,1,0,1,true,5,0.0,Dict(:check_every=>4,:max_iter=>40))
    worker=parse_model_worker_args(collect(common_worker_args(request)))
    @test worker.kwargs==request.kwargs
    @test isempty(parse_model_worker_args(collect(common_worker_args(WorkerRequest(1,1,"cg","Float64",17,1,1,0,1,true,5,0.0)))).kwargs)
    @test_throws MethodError build_benchmark(ConjugateGradientBenchmark,Float64,17,1;typo=4)
    mktemp() do path,io
        raw["cg"][1]["fusion"]=[true,false]
        other=deepcopy(raw["cg"][1]); other["kwargs"]["check_every"]=4
        push!(raw["cg"],other)
        TOML.print(io,raw); close(io)
        gs,specs=parse_config(path)
        runs=plan_runs(specs,gs,raw,Dict(),10^12)
        @test count(r->r.model==:jacc,runs)==2 # one per kwargs, despite fusion sweep
        @test count(r->r.model==:cunumeric && r.spec.name=="cg",runs)==4
        @test length(unique(results_subdir(r.spec) for r in runs))==2
    end
    @test !supports_benchmark(JACCModel(),"cg_accelerated")
    @test supports_run(JACCModel(),"cg",2)
    @test_throws ErrorException memory_estimate(
        ConjugateGradientBenchmark{Float64}(;N=17),MemoryContext(;model=:jacc,gpus=2))
    @test peak_bytes(memory_estimate(
        ConjugateGradientBenchmark{Float64}(;N=16),MemoryContext(;model=:jacc,gpus=2))) > 0
    @test !supports_benchmark(CUDAJLModel(),"cg")
    @test !supports_benchmark(CuPyNumericModel(),"cg")
    @test !supports_benchmark(DaggerModel(),"cg")
end
