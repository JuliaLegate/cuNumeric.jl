using Test, Statistics, TOML
include("../src/core.jl")
include_benchmarks()
include("../src/models.jl")
include("../src/parse_benchmarks.jl")
include("../src/memory.jl")
include("../src/planning.jl")
include("../src/runner.jl")
include("../src/result_rows.jl")
include("timing.jl")

const CONFIG = joinpath(@__DIR__, "..", "benchmarks.toml")
const RAW = TOML.parsefile(CONFIG)
const GROUPS = parse_plot_groups(CONFIG)

# A CPU array that counts materialized arrays, exercising Julia's actual
# broadcast lowering rather than checking the kernel's source spelling.
const MATERIALIZATIONS = Ref(0)
struct CountedArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
struct CountedStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
CountedStyle{N}(::Val{M}) where {N,M} = CountedStyle{M}()
Base.size(a::CountedArray) = size(a.data)
Base.getindex(a::CountedArray, I...) = getindex(a.data, I...)
Base.setindex!(a::CountedArray, v, I...) = setindex!(a.data, v, I...)
Base.BroadcastStyle(::Type{<:CountedArray{T,N}}) where {T,N} = CountedStyle{N}()
function Base.similar(bc::Base.Broadcast.Broadcasted{CountedStyle{N}}, ::Type{T}) where {N,T}
    MATERIALIZATIONS[] += 1
    return CountedArray(Array{T}(undef, map(length, axes(bc))))
end
function Base.similar(a::CountedArray, ::Type{T}, dims::Dims) where {T}
    MATERIALIZATIONS[] += 1
    return CountedArray(Array{T}(undef, dims))
end

@testset "Monte Carlo broadcasts fuse across negation" begin
    for T in (Float32, Float64)
        data = T[0, 0.5, 1, 2, 5]
        x = CountedArray(data)
        b = MonteCarloIntegration{T}(; n_samples=length(x))
        MATERIALIZATIONS[] = 0
        got = run!(b, x)
        @test MATERIALIZATIONS[] == 1
        @test got ≈ (T(10)/length(data))*sum(exp(-v^2) for v in data)
        # Reproduce the old expression to prove this test detects the bug.
        MATERIALIZATIONS[] = 0
        sum(exp.(-x .^ 2))
        @test MATERIALIZATIONS[] == 3
    end
end

@testset "cuPyNumeric preflight" begin
    gs = GlobalSettings(; n_warmup=1, n_iter=1, models=[:cunumeric, :cupynumeric])
    s = BenchmarkSpec(
        "montecarlo", "Float32", 1, 8, true, gs.models, 1, 1, 1, [0, 0], true, nothing, nothing
    )
    runs = plan_runs([s], gs, RAW, GROUPS, 1000000)
    env = Dict("CUNUMERIC_BENCH_CONDA"=>"/test/conda", "CUPYNUMERIC_ENV"=>"testenv")
    @test_throws ErrorException preflight_models(runs; env, which=x->nothing)
    @test_throws ErrorException preflight_models(runs; env, which=identity, check=c->false)
    @test preflight_models(runs; env, which=identity, check=c->true) === nothing
end

function spec(
    name; T="Float32", gpus=1, fusion=true, models=[:cunumeric], N=nothing, M=nothing, auto=true
)
    return BenchmarkSpec(name, T, gpus, 8, fusion, collect(models), 2, 5, 2,
        auto ? [0, 0] : [N, M], auto, N, M)
end

@testset "Configuration and CLI" begin
    gs, ss = parse_config(CONFIG; only="grayscott", fusion_override=[true, false])
    @test all(startswith(s.name, "grayscott") for s in ss)
    @test Set(s.fusion for s in ss)==Set([true, false])
    @test_throws ErrorException parse_config(CONFIG; only="missing")
    o = cli_options(["--only=montecarlo", "--fusion=both", "--dry-run"])
    @test o.only == "montecarlo" && o.dry && o.fusion == [true, false]
    @test cli_options(["--models=jacc,dagger"]).models == [:jacc, :dagger]
    @test_throws ErrorException cli_options(["--models=missing"])
    @test_throws ErrorException cli_options(["--typo"])
    p = positional_spec(["1", "8", "montecarlo", "Float32", "auto", "1", "5", "2", "2"], gs)
    @test p.autosize && p.M_hint==1 && p.n_iter==5
    @test main(["--only=montecarlo", "--dry-run"];
        budget_provider=(f, p)->(1_000_000, f),
        executor=(args...)->error("dry-run launched workers"))==0
end

@testset "Execution model registry and isolation" begin
    @test parse_models(["cuNumeric", "CUDA.jl", "JACC", "Dagger.jl"]) ==
        [:cunumeric, :cudajl, :jacc, :dagger]
    @test supports_benchmark(execution_model(:jacc), "montecarlo")
    @test !supports_benchmark(execution_model(:jacc), "gemm")
    @test !supports_gpu_count(execution_model(:cudajl), 2)

    gs, specs = parse_config(CONFIG; only="montecarlo", models_override=[:jacc, :dagger])
    runs = plan_runs(specs, gs, RAW, GROUPS, 1_000_000)
    @test Set(r.model for r in runs) == Set((:jacc, :dagger))
    @test all(r.model != :cunumeric for r in runs)

    request = WorkerRequest(2, 8, "montecarlo", "Float32", 1024, 1, 3, 1, 2, false, 5, 1024.0)
    one_gpu_request = WorkerRequest(
        1, 8, "montecarlo", "Float32", 1024, 1, 3, 1, 2, false, 5, 1024.0
    )
    cunumeric_cmd = join(
        wrapped_worker_command(execution_model(:cunumeric), one_gpu_request, pwd()).exec, ' '
    )
    cuda_cmd = join(
        wrapped_worker_command(execution_model(:cudajl), one_gpu_request, pwd()).exec, ' '
    )
    jacc_cmd = join(wrapped_worker_command(execution_model(:jacc), request, pwd()).exec, ' ')
    dagger_cmd = join(wrapped_worker_command(execution_model(:dagger), request, pwd()).exec, ' ')
    @test occursin(joinpath("src", "cunumeric", "single.jl"), cunumeric_cmd)
    @test occursin("--project=$(joinpath(pwd(),"environments","cunumeric"))", cunumeric_cmd)
    @test occursin(joinpath("src", "cuda", "single.jl"), cuda_cmd)
    @test occursin("--project=$(joinpath(pwd(),"environments","cuda"))", cuda_cmd)
    @test occursin(joinpath("src", "jacc", "single.jl"), jacc_cmd)
    @test occursin("--project=$(joinpath(pwd(),"environments","jacc"))", jacc_cmd)
    @test occursin(joinpath("src", "dagger", "single.jl"), dagger_cmd)
    @test occursin("--project=$(joinpath(pwd(),"environments","dagger"))", dagger_cmd)

    @test selected_cuda_visibility(2; env=Dict{String,String}()) == "0,1"
    scheduler_env = Dict("CUDA_VISIBLE_DEVICES"=>"GPU-a, MIG-b, 7")
    @test selected_cuda_visibility(2; env=scheduler_env) == "GPU-a,MIG-b"
    @test_throws ErrorException selected_cuda_visibility(
        2; env=Dict("CUDA_VISIBLE_DEVICES"=>"GPU-a")
    )
    @test_throws ErrorException selected_cuda_visibility(
        1; env=Dict("CUDA_VISIBLE_DEVICES"=>"")
    )

    cunumeric_project = read(
        joinpath(@__DIR__, "..", "environments", "cunumeric", "Project.toml"), String
    )
    cuda_project = read(
        joinpath(@__DIR__, "..", "environments", "cuda", "Project.toml"), String
    )
    @test occursin("cuNumeric =", cunumeric_project)
    @test !occursin("JACC =", cunumeric_project)
    @test !occursin("Dagger =", cunumeric_project)
    @test occursin("CUDA =", cuda_project)
    @test !occursin("cuNumeric =", cuda_project)
    @test !occursin("JACC =", cuda_project)
    @test !occursin("Dagger =", cuda_project)

    jacc_manifest = read(joinpath(@__DIR__, "..", "environments", "jacc", "Manifest.toml"), String)
    dagger_manifest = read(
        joinpath(@__DIR__, "..", "environments", "dagger", "Manifest.toml"), String
    )
    @test occursin("[[deps.JACC]]", jacc_manifest)
    @test !occursin("[[deps.Dagger]]", jacc_manifest)
    @test !occursin("[[deps.cuNumeric]]", jacc_manifest)
    @test occursin("[[deps.Dagger]]", dagger_manifest)
    @test !occursin("[[deps.JACC]]", dagger_manifest)
    @test !occursin("[[deps.cuNumeric]]", dagger_manifest)

    runner = joinpath(@__DIR__, "..", "run_benchmark.sh")
    ok = `bash $runner --model=jacc --gpus=1 --cpus=0 -- bash -c $("test \"\$CUNUMERIC_BENCH_ACTIVE_MODEL\" = jacc")`
    nested = addenv(`bash $runner --model=jacc --gpus=1 --cpus=0 -- true`,
        "CUNUMERIC_BENCH_ACTIVE_MODEL"=>"cunumeric")
    @test success(pipeline(ok; stdout=devnull, stderr=devnull))
    @test !success(pipeline(nested; stdout=devnull, stderr=devnull))
end

@testset "Memory dispatch matrix" begin
    for T in (Float32, Float64),
        fusion in (false, true),
        model in (:cunumeric, :cudajl, :cupynumeric)

        c = MemoryContext(; model, fusion, workspace_bytes=0)
        for (name, B) in BENCHMARKS
            endswith(name, "_accelerated") && model != :cunumeric && continue
            m = if B <: AbstractDMD
                16
            elseif B <: AbstractGrayScott || B <: GEMM
                64
            else
                1
            end
            b = build_benchmark(B, T, 64, m)
            estimate = memory_estimate(b, c)
            @test estimate.initialization>0 && estimate.iteration>0
            @test !isempty(estimate.explanation)
        end
    end
    b = MonteCarloIntegration{Float32}(; n_samples=1024)
    @test peak_bytes(memory_estimate(b, MemoryContext())) == 8192
    @test peak_bytes(memory_estimate(b, MemoryContext(; model=:cupynumeric))) == 12288
    @test peak_bytes(memory_estimate(b, MemoryContext(; fusion=false))) == 16384
    @test_throws ErrorException memory_estimate(GEMM{Float32}(; N=64, M=64), MemoryContext())
    @test_throws ErrorException memory_estimate(b, MemoryContext(; model=:cudajl, gpus=2))
    d = DMDBaseline{Float32}(; N=1024, M=16)
    @test peak_bytes(memory_estimate(d, MemoryContext(; gpus=1, workspace_bytes=0))) ==
        peak_bytes(memory_estimate(d, MemoryContext(; gpus=8, workspace_bytes=0)))
end

@testset "Shared sweep planning" begin
    gs = GlobalSettings(; n_warmup=2, n_iter=5)
    all_models = [:cunumeric, :cudajl, :cupynumeric]
    ss = [
        spec("montecarlo"; gpus=p, fusion=f, models=all_models) for p in (1, 2, 4, 8) for
        f in (true, false)
    ]
    runs = plan_runs(ss, gs, RAW, GROUPS, 1_000_000)
    @test length(runs)==8+4+1
    for p in (1, 2, 4, 8)
        @test length(unique((r.N, r.M) for r in runs if r.spec.gpus==p))==1
    end
    @test all(peak_bytes(r.memory)<=1_000_000 for r in runs)
    @test maximum(r.N for r in runs)==8minimum(r.N for r in runs)
    fused = plan_runs([spec("montecarlo"; models=all_models)], gs, RAW, GROUPS, 1_000_000)
    @test first(fused).N >= first(runs).N
    onlyoff = plan_runs(
        [spec("montecarlo"; fusion=false, models=all_models)], gs, RAW, GROUPS, 1_000_000
    )
    @test any(r.model==:cupynumeric for r in onlyoff)
    @test_throws ErrorException plan_runs(
        [spec("montecarlo"; N=1000000, M=1, auto=false)], gs, RAW, GROUPS, 100
    )
    @test_throws ErrorException plan_runs(
        [spec("montecarlo"; N=8, M=1, auto=false), spec("montecarlo"; N=16, M=1, auto=false)],
        gs,
        RAW,
        GROUPS,
        100000,
    )
    @test_throws ErrorException plan_runs([spec("montecarlo"; gpus=0)], gs, RAW, GROUPS, 100000)
    raw = deepcopy(RAW)
    raw["workspace"] = Dict("dmd_baseline"=>Dict("cunumeric"=>0))
    single = plan_runs(
        [spec("dmd_baseline"; M=16)],
        GlobalSettings(; n_warmup=1, n_iter=1),
        raw,
        GROUPS,
        10_000_000,
    )
    sweep = plan_runs(
        [spec("dmd_baseline"; M=16, gpus=p) for p in (1, 8)],
        GlobalSettings(; n_warmup=1, n_iter=1),
        raw,
        GROUPS,
        10_000_000,
    )
    @test first(sweep).N < first(single).N
    @test all(peak_bytes(r.memory)<=10_000_000 for r in sweep)
end

@testset "GPU budgets" begin
    inventory = "0, GPU-aaa, 1000, 900\n1, GPU-bbb, 2000, 1900\n"
    kw = (; inventory, fraction="0.75", fbmem=nothing)
    @test first(selected_gpu_budget(0.75, 1; kw..., visibility="GPU-bbb")) == 1500*1024^2
    @test first(selected_gpu_budget(0.75, 2; kw..., visibility="1,0")) == 750*1024^2
    @test_throws ErrorException selected_gpu_budget(0.75, 2; kw..., visibility="1")
    @test_throws ErrorException selected_gpu_budget(0.75, 1; kw..., visibility="")
    @test_throws ErrorException selected_gpu_budget(0.75, 1; kw..., visibility="2")
end

@testset "Result dimension isolation" begin
    rows = [Row(1, 32, 1, 1.0, 2.0), Row(1, 32, 1, 3.0, 4.0)]
    a = aggregate(rows)
    @test only(a).N==32 && only(a).t==2
    @test_throws ErrorException aggregate(vcat(rows, [Row(1, 64, 1, 1.0, 2.0)]))
    @test_throws ErrorException validate_series_sizes([
        (agg=a,), (agg=aggregate([Row(1, 64, 1, 1.0, 2.0)]),)
    ])
end

@testset "Variant lifetimes and rectangular constraints" begin
    baseline = GrayScottBaseline{Float32}(; N=64, M=32)
    accelerated = GrayScottFunctionAccelerated{Float32}(; N=64, M=32)
    for f in (true, false)
        c = MemoryContext(; fusion=f, steps=10)
        @test peak_bytes(memory_estimate(accelerated, c)) < peak_bytes(memory_estimate(baseline, c))
    end
    @test peak_bytes(memory_estimate(baseline, MemoryContext(; fusion=true))) <
        peak_bytes(memory_estimate(baseline, MemoryContext(; fusion=false)))
    @test estimate_scaling(GEMM{Float32}(; N=64, M=32), 8)==(128, 64)
    @test estimate_scaling(baseline, 4)==(128, 64)
    @test_throws ErrorException memory_estimate(baseline, MemoryContext(; steps=0))
    @test_throws ErrorException memory_estimate(accelerated, MemoryContext(; model=:cupynumeric))
    gs = GlobalSettings(; n_warmup=1, n_iter=1)
    ss = [
        spec(n; gpus=p, fusion=f) for n in ("grayscott_baseline", "grayscott_function_accelerated")
        for p in (1, 4) for f in (false, true)
    ]
    runs = plan_runs(ss, gs, RAW, GROUPS, 10_000_000)
    for p in (1, 4)
        @test length(unique((r.N, r.M) for r in runs if r.spec.gpus==p))==1
    end
end

@testset "Execution isolation and failure status" begin
    gs = GlobalSettings(; n_warmup=1, n_iter=1)
    runs = plan_runs(
        [spec("montecarlo"; T=T) for T in ("Float32", "Float64")], gs, RAW, GROUPS, 1_000_000
    )
    opts = cli_options(["--only=montecarlo"])
    mktempdir() do dir
        calls = Cmd[]
        launch(cmd) = (push!(calls, cmd); nothing)
        @test execute_plan(
            runs, gs, opts, 1_000_000, RAW; launch, prepare=(f, v)->nothing,
            results_root=dir, preflight=runs->nothing,
        )==0
        @test length(calls)==4 # two workers and one plot per dtype
        run_dir = only(readdir(dir; join=true))
        manifest = TOML.parsefile(joinpath(run_dir, "manifest.toml"))
        @test manifest["status"]=="complete"
        @test all(r["status"]=="complete" for r in manifest["runs"])
        @test occursin("Float32", join(calls[1].env))
        @test occursin("Float64", join(calls[2].env))
        @test any(occursin("plot_results.jl", join(c.exec)) for c in calls)
    end
    mktempdir() do dir
        calls = Ref(0)
        function fail_first(cmd)
            calls[] += 1
            calls[] == 1 && error("simulated worker failure")
        end
        @test execute_plan(
            runs, gs, opts, 1_000_000, RAW; launch=fail_first,
            prepare=(f, v)->nothing, results_root=dir, preflight=runs->nothing,
        )==1
        @test calls[]==4 # no retry, later worker and plots still run
        manifest = TOML.parsefile(joinpath(only(readdir(dir; join=true)), "manifest.toml"))
        @test manifest["status"]=="incomplete"
        @test manifest["runs"][1]["status"]=="failed"
        @test manifest["runs"][2]["status"]=="complete"
    end
end
