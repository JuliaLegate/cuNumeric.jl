using Test, Statistics, TOML, LinearAlgebra
include("../src/core.jl")
include_benchmarks()
include("../src/cunumeric/benchmarks/grayscott_accelerate_forms.jl")
include("../src/models.jl")
include("../src/parse_benchmarks.jl")
include("../src/memory.jl")
include("../src/planning.jl")
include("../src/runner.jl")
include("../src/result_rows.jl")
include("../src/model_worker.jl")
include("timing.jl")
include("nas.jl")

const CONFIG = joinpath(@__DIR__, "..", "benchmarks.toml")
const SMOKE_CONFIG = joinpath(@__DIR__, "..", "benchmarks_smoke.toml")
const GRAYSCOTT_MULTIGPU_CONFIG = joinpath(@__DIR__, "..", "benchmarks_grayscott_multigpu.toml")
const FORMS_CONFIG = joinpath(@__DIR__, "..", "benchmarks_grayscott_forms.toml")
const RAW = TOML.parsefile(CONFIG)
const GROUPS = parse_plot_groups(CONFIG)
const FORMS_GROUPS = parse_plot_groups(FORMS_CONFIG)

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

@testset "CPU correctness inputs" begin
    mc = MonteCarloIntegration{Float32}(; n_samples=2048)
    mc_check = correctness_problem(mc)
    @test correctness_uses_cpu(mc)
    @test dims(mc_check) == (1024, 1)
    @test only(correctness_seed(mc_check)) ==
        Float32.(range(0.0f0, 10.0f0; length=1024))

    gemm = GEMM{Float32}(; N=16, M=12)
    gemm_check = correctness_problem(gemm)
    C, A, B = correctness_seed(gemm_check)
    @test correctness_uses_cpu(gemm)
    @test dims(gemm_check) == (8, 8)
    @test C == zeros(Float32, 8, 8)
    @test run!(gemm_check, C, A, B) ≈ A * B
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
    @test BENCHMARKS["grayscott"] === GrayScottAccelerated
    @test BENCHMARKS["grayscott_plain"] === GrayScottBaseline
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

@testset "Smoke configuration" begin
    gs, specs = parse_config(SMOKE_CONFIG)
    @test gs.models == [:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger]
    @test gs.check_correctness
    @test !isempty(specs) && all(spec.name in keys(BENCHMARKS) for spec in specs)
    @test all(spec.models == gs.models for spec in specs)
end

@testset "Execution model registry and isolation" begin
    @test parse_models(["cuNumeric", "CUDA.jl", "JACC", "Dagger.jl"]) ==
        [:cunumeric, :cudajl, :jacc, :dagger]
    @test supports_benchmark(execution_model(:jacc), "montecarlo")
    @test supports_benchmark(execution_model(:jacc), "gemm")
    @test supports_benchmark(execution_model(:dagger), "gemm")
    @test !supports_gpu_count(execution_model(:cudajl), 2)
    @test !supports_run(execution_model(:jacc), "grayscott", 2)
    @test supports_run(execution_model(:dagger), "grayscott", 2)

    gs_gray, specs_gray = parse_config(GRAYSCOTT_MULTIGPU_CONFIG)
    runs_gray = plan_runs(
        specs_gray, gs_gray, TOML.parsefile(GRAYSCOTT_MULTIGPU_CONFIG), Dict(), 10^12
    )
    @test Set(r.model for r in runs_gray) == Set((:cunumeric, :cupynumeric, :dagger))
    @test Set(r.spec.gpus for r in runs_gray) == Set((1, 2, 4, 8))
    gray_sizes = Dict(1=>24000, 2=>33944, 4=>48000, 8=>67888)
    @test all(
        (r.N, r.M) == (gray_sizes[r.spec.gpus], gray_sizes[r.spec.gpus]) for
        r in runs_gray
    )

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
    clean_library_path = addenv(
        `bash $runner --model=cudajl --gpus=1 --cpus=0 -- bash -c $("test -z \"\${LD_LIBRARY_PATH+x}\"")`,
        "LD_LIBRARY_PATH"=>"/system/cuda/lib64",
    )
    @test success(pipeline(ok; stdout=devnull, stderr=devnull))
    @test !success(pipeline(nested; stdout=devnull, stderr=devnull))
    @test success(pipeline(clean_library_path; stdout=devnull, stderr=devnull))

    visibility_probe = `bash -c $("printf %s \"\$CUDA_VISIBLE_DEVICES\"")`
    default_visibility = `env -u CUDA_VISIBLE_DEVICES bash $runner \
        --model=dagger --gpus=2 --cpus=0 -- $visibility_probe`
    scheduler_visibility = addenv(
        `bash $runner --model=dagger --gpus=2 --cpus=0 -- $visibility_probe`,
        "CUDA_VISIBLE_DEVICES"=>"GPU-a,MIG-b,7",
    )
    @test read(default_visibility, String) == "0,1"
    @test read(scheduler_visibility, String) == "GPU-a,MIG-b"
end

@testset "Memory dispatch matrix" begin
    for T in (Float32, Float64),
        fusion in (false, true),
        model in (:cunumeric, :cudajl, :cupynumeric)

        c = MemoryContext(; model, fusion, workspace_bytes=0)
        for (name, B) in BENCHMARKS
            supports_benchmark(execution_model(model), name) || continue
            B <: NASFourierTransform && continue
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
    # GEMM resolves to a source default; jacc's kernel needs no cuBLAS scratch.
    @test memory_estimate(GEMM{Float32}(; N=64, M=64), MemoryContext()).workspace ==
        CUBLAS_WORKSPACE_PER_GPU
    @test memory_estimate(GEMM{Float32}(; N=64, M=64), MemoryContext(; model=:jacc)).workspace == 0
    # A native benchmark without a source default still requires an explicit bound.
    @test_throws ErrorException memory_estimate(DMDBaseline{Float32}(; N=64, M=16), MemoryContext())
    @test_throws ErrorException memory_estimate(b, MemoryContext(; model=:cudajl, gpus=2))
    d = DMDBaseline{Float32}(; N=1024, M=16)
    @test peak_bytes(memory_estimate(d, MemoryContext(; gpus=1, workspace_bytes=0))) ==
        peak_bytes(memory_estimate(d, MemoryContext(; gpus=8, workspace_bytes=0)))
    cg = ConjugateGradientAccelerated{Float64}(; N=9_000_000)
    cg1 = peak_bytes(memory_estimate(cg, MemoryContext(; steps=1)))
    @test cg1 == 24 * 9_000_000 * sizeof(Float64)
    @test peak_bytes(memory_estimate(cg, MemoryContext(; steps=1001))) == cg1
    cg2 = ConjugateGradientAccelerated{Float64}(; N=18_000_000)
    @test peak_bytes(memory_estimate(cg2, MemoryContext(; gpus=2))) == cg1
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
    # A pinned oversize is respected (kept, warns) so the user can force an OOM.
    oversize = @test_logs (:warn,) match_mode=:any plan_runs(
        [spec("montecarlo"; N=1000000, M=1, auto=false)], gs, RAW, GROUPS, 100
    )
    @test only(oversize).N == 1000000
    @test peak_bytes(only(oversize).memory) > 100
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
    default_accelerated = GrayScottAccelerated{Float32}(; N=64, M=32)
    accelerated = GrayScottFunctionAccelerated{Float32}(; N=64, M=32)
    for f in (true, false)
        c = MemoryContext(; fusion=f, steps=10)
        @test peak_bytes(memory_estimate(accelerated, c)) ==
            peak_bytes(memory_estimate(baseline, c))
        @test peak_bytes(memory_estimate(default_accelerated, c)) ==
            peak_bytes(memory_estimate(accelerated, c))
    end
    @test peak_bytes(memory_estimate(baseline, MemoryContext(; steps=100))) ==
        peak_bytes(memory_estimate(baseline, MemoryContext(; steps=1)))
    @test peak_bytes(memory_estimate(baseline, MemoryContext(; fusion=true))) <
        peak_bytes(memory_estimate(baseline, MemoryContext(; fusion=false)))
    # The grid is tiled per GPU: more GPUs at fixed N means less memory per GPU,
    # and a weak-scaled aligned (N, GPUs) pair holds per-GPU memory ~constant.
    big1 = GrayScottFunctionAccelerated{Float32}(; N=4096, M=4096)
    @test peak_bytes(memory_estimate(big1, MemoryContext(; gpus=4))) <
        peak_bytes(memory_estimate(big1, MemoryContext(; gpus=1)))
    p1 = peak_bytes(memory_estimate(big1, MemoryContext(; gpus=1)))
    big4 = GrayScottFunctionAccelerated{Float32}(; N=8192, M=8192) # N*2 for 4 GPUs (2D)
    p4 = peak_bytes(memory_estimate(big4, MemoryContext(; gpus=4)))
    @test 0.9 < Float64(p4)/Float64(p1) < 1.1
    @test estimate_scaling(GEMM{Float32}(; N=64, M=32), 8)==(128, 64)
    @test estimate_scaling(baseline, 4)==(128, 64)
    @test_throws ErrorException memory_estimate(baseline, MemoryContext(; steps=0))
    @test_throws ErrorException memory_estimate(accelerated, MemoryContext(; model=:cupynumeric))
    gs = GlobalSettings(; n_warmup=1, n_iter=1)
    ss = [
        spec(n; gpus=p, fusion=f) for n in ("grayscott_plain", "grayscott_function_accelerated")
        for p in (1, 4) for f in (false, true)
    ]
    runs = plan_runs(ss, gs, RAW, FORMS_GROUPS, 10_000_000)
    for p in (1, 4)
        @test length(unique((r.N, r.M) for r in runs if r.spec.gpus==p))==1
    end
    gsx = GlobalSettings(; n_warmup=1, n_iter=20)
    sx = [spec(n; gpus=1) for n in ("grayscott_plain", "grayscott_function_accelerated")]
    rx = plan_runs(sx, gsx, RAW, FORMS_GROUPS, 10_000_000)
    @test length(unique((r.N, r.M) for r in rx))==1
    accel = only(r for r in rx if r.spec.name=="grayscott_function_accelerated")
    base = only(r for r in rx if r.spec.name=="grayscott_plain")
    @test peak_bytes(accel.memory) <= accel.budget
    @test peak_bytes(base.memory) <= base.budget
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

include("cg.jl")
