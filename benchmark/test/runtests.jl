using Test, Statistics, TOML
include("../src/core.jl")
include_benchmarks()
include("../src/parse_benchmarks.jl")
include("../src/memory.jl")
include("../src/planning.jl")
include("../src/runner.jl")
include("../src/result_rows.jl")

const CONFIG = joinpath(@__DIR__,"..","benchmarks.toml")
const RAW = TOML.parsefile(CONFIG)
const GROUPS = parse_plot_groups(CONFIG)

function spec(name;T="Float32",gpus=1,fusion=true,cuda=false,N=nothing,M=nothing,auto=true)
    BenchmarkSpec(name,T,gpus,8,fusion,cuda,2,5,2,
        auto ? [0,0] : [N,M],auto,N,M)
end

@testset "Configuration and CLI" begin
    gs,ss = parse_config(CONFIG;only="grayscott",fusion_override=[true,false])
    @test all(startswith(s.name,"grayscott") for s in ss)
    @test Set(s.fusion for s in ss)==Set([true,false])
    @test_throws ErrorException parse_config(CONFIG;only="missing")
    o = cli_options(["--only=montecarlo","--fusion=both","--dry-run"])
    @test o.only == "montecarlo" && o.dry && o.fusion == [true,false]
    @test_throws ErrorException cli_options(["--typo"])
    p = positional_spec(["1","8","montecarlo","Float32","auto","1","5","2","2"],gs)
    @test p.autosize && p.M_hint==1 && p.n_iter==5
    @test main(["--only=montecarlo","--dry-run"];
        budget_provider=(f,p)->(1_000_000,f),executor=(args...)->error("dry-run launched workers"))==0
end

@testset "Memory dispatch matrix" begin
    for T in (Float32,Float64), fusion in (false,true), backend in (:cunumeric,:cudajl,:cupynumeric)
        c = MemoryContext(;backend,fusion,workspace_bytes=0)
        for (name,B) in BENCHMARKS
            endswith(name,"_accelerated") && backend != :cunumeric && continue
            m = B <: AbstractDMD ? 16 : B <: AbstractGrayScott || B <: GEMM ? 64 : 1
            b = build_benchmark(B,T,64,m)
            estimate = memory_estimate(b,c)
            @test estimate.initialization>0 && estimate.iteration>0
            @test !isempty(estimate.explanation)
        end
    end
    b = MonteCarloIntegration{Float32}(;n_samples=1024)
    @test peak_bytes(memory_estimate(b,MemoryContext())) == 8192
    @test peak_bytes(memory_estimate(b,MemoryContext(;backend=:cupynumeric))) == 12288
    @test peak_bytes(memory_estimate(b,MemoryContext(;fusion=false))) == 16384
    @test_throws ErrorException memory_estimate(GEMM{Float32}(;N=64,M=64),MemoryContext())
    @test_throws ErrorException memory_estimate(b,MemoryContext(;backend=:cudajl,gpus=2))
    d = DMDBaseline{Float32}(;N=1024,M=16)
    @test peak_bytes(memory_estimate(d,MemoryContext(;gpus=1,workspace_bytes=0))) ==
        peak_bytes(memory_estimate(d,MemoryContext(;gpus=8,workspace_bytes=0)))
end

@testset "Shared sweep planning" begin
    gs = GlobalSettings(;n_warmup=2,n_iter=5,cupynumeric=true)
    ss = [spec("montecarlo";gpus=p,fusion=f,cuda=true) for p in (1,2,4,8) for f in (true,false)]
    runs = plan_runs(ss,gs,RAW,GROUPS,1_000_000)
    @test length(runs)==8+4+1
    for p in (1,2,4,8)
        @test length(unique((r.N,r.M) for r in runs if r.spec.gpus==p))==1
    end
    @test all(peak_bytes(r.memory)<=1_000_000 for r in runs)
    @test maximum(r.N for r in runs)==8minimum(r.N for r in runs)
    fused = plan_runs([spec("montecarlo")],gs,RAW,GROUPS,1_000_000)
    @test first(fused).N >= first(runs).N
    onlyoff = plan_runs([spec("montecarlo";fusion=false)],gs,RAW,GROUPS,1_000_000)
    @test any(r.backend==:cupynumeric for r in onlyoff)
    @test_throws ErrorException plan_runs([spec("montecarlo";N=1000000,M=1,auto=false)],gs,RAW,GROUPS,100)
    @test_throws ErrorException plan_runs([spec("montecarlo";N=8,M=1,auto=false),spec("montecarlo";N=16,M=1,auto=false)],gs,RAW,GROUPS,100000)
    @test_throws ErrorException plan_runs([spec("montecarlo";gpus=0)],gs,RAW,GROUPS,100000)
    raw = deepcopy(RAW)
    raw["workspace"] = Dict("dmd_baseline"=>Dict("cunumeric"=>0))
    single = plan_runs([spec("dmd_baseline";M=16)],GlobalSettings(;n_warmup=1,n_iter=1),raw,GROUPS,10_000_000)
    sweep = plan_runs([spec("dmd_baseline";M=16,gpus=p) for p in (1,8)],GlobalSettings(;n_warmup=1,n_iter=1),raw,GROUPS,10_000_000)
    @test first(sweep).N < first(single).N
    @test all(peak_bytes(r.memory)<=10_000_000 for r in sweep)
end

@testset "GPU budgets" begin
    inventory = "0, GPU-aaa, 1000, 900\n1, GPU-bbb, 2000, 1900\n"
    kw = (;inventory,fraction="0.75",fbmem=nothing)
    @test first(selected_gpu_budget(.75,1;kw...,visibility="GPU-bbb")) == 1500*1024^2
    @test first(selected_gpu_budget(.75,2;kw...,visibility="1,0")) == 750*1024^2
    @test_throws ErrorException selected_gpu_budget(.75,2;kw...,visibility="1")
    @test_throws ErrorException selected_gpu_budget(.75,1;kw...,visibility="")
    @test_throws ErrorException selected_gpu_budget(.75,1;kw...,visibility="2")
end

@testset "Result dimension isolation" begin
    rows = [Row(1,32,1,1.,2.),Row(1,32,1,3.,4.)]
    a = aggregate(rows)
    @test only(a).N==32 && only(a).t==2
    @test_throws ErrorException aggregate(vcat(rows,[Row(1,64,1,1.,2.)]))
    @test_throws ErrorException validate_series_sizes([(agg=a,),(agg=aggregate([Row(1,64,1,1.,2.)]),)])
end

@testset "Variant lifetimes and rectangular constraints" begin
    baseline = GrayScottBaseline{Float32}(;N=64,M=32)
    accelerated = GrayScottFunctionAccelerated{Float32}(;N=64,M=32)
    for f in (true,false)
        c = MemoryContext(;fusion=f,steps=10)
        @test peak_bytes(memory_estimate(accelerated,c)) < peak_bytes(memory_estimate(baseline,c))
    end
    @test peak_bytes(memory_estimate(baseline,MemoryContext(;fusion=true))) <
        peak_bytes(memory_estimate(baseline,MemoryContext(;fusion=false)))
    @test estimate_scaling(GEMM{Float32}(;N=64,M=32),8)==(128,64)
    @test estimate_scaling(baseline,4)==(128,64)
    @test_throws ErrorException memory_estimate(baseline,MemoryContext(;steps=0))
    @test_throws ErrorException memory_estimate(accelerated,MemoryContext(;backend=:cupynumeric))
    gs = GlobalSettings(;n_warmup=1,n_iter=1)
    ss = [spec(n;gpus=p,fusion=f) for n in ("grayscott_baseline","grayscott_function_accelerated") for p in (1,4) for f in (false,true)]
    runs = plan_runs(ss,gs,RAW,GROUPS,10_000_000)
    for p in (1,4)
        @test length(unique((r.N,r.M) for r in runs if r.spec.gpus==p))==1
    end
end

@testset "Execution isolation and failure status" begin
    gs = GlobalSettings(;n_warmup=1,n_iter=1)
    runs = plan_runs([spec("montecarlo";T=T) for T in ("Float32","Float64")],gs,RAW,GROUPS,1_000_000)
    opts = cli_options(["--only=montecarlo"])
    mktempdir() do dir
        calls = Cmd[]
        launch(cmd) = (push!(calls,cmd); nothing)
        @test execute_plan(runs,gs,opts,1_000_000,RAW;launch,prepare=(f,v)->nothing,results_root=dir)==0
        @test length(calls)==4 # two workers and one plot per dtype
        run_dir = only(readdir(dir;join=true))
        manifest = TOML.parsefile(joinpath(run_dir,"manifest.toml"))
        @test manifest["status"]=="complete"
        @test all(r["status"]=="complete" for r in manifest["runs"])
        @test occursin("Float32",join(calls[1].env))
        @test occursin("Float64",join(calls[2].env))
        @test any(occursin("plot_results.jl",join(c.exec)) for c in calls)
    end
    mktempdir() do dir
        calls = Ref(0)
        function fail_first(cmd)
            calls[] += 1
            calls[] == 1 && error("simulated worker failure")
        end
        @test execute_plan(runs,gs,opts,1_000_000,RAW;launch=fail_first,prepare=(f,v)->nothing,results_root=dir)==1
        @test calls[]==4 # no retry, later worker and plots still run
        manifest = TOML.parsefile(joinpath(only(readdir(dir;join=true)),"manifest.toml"))
        @test manifest["status"]=="incomplete"
        @test manifest["runs"][1]["status"]=="failed"
        @test manifest["runs"][2]["status"]=="complete"
    end
end
