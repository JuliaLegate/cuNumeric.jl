# Execution-model registry for the orchestrator.
#
# This file deliberately contains no `using cuNumeric`, `using JACC`, or
# `using Dagger`.  Model packages belong to their dedicated worker processes;
# the orchestrator only needs enough metadata to plan and launch those workers.

using TOML

abstract type ExecutionModel end

struct CuNumericModel <: ExecutionModel end
struct CuPyNumericModel <: ExecutionModel end
struct CUDAJLModel <: ExecutionModel end
struct JACCModel <: ExecutionModel end
struct DaggerModel <: ExecutionModel end

const EXECUTION_MODELS = ExecutionModel[
    CuNumericModel(),
    CuPyNumericModel(),
    CUDAJLModel(),
    JACCModel(),
    DaggerModel(),
]

model_id(::CuNumericModel) = :cunumeric
model_id(::CuPyNumericModel) = :cupynumeric
model_id(::CUDAJLModel) = :cudajl
model_id(::JACCModel) = :jacc
model_id(::DaggerModel) = :dagger

model_label(::CuNumericModel) = "cuNumeric.jl"
model_label(::CuPyNumericModel) = "cuPyNumeric"
model_label(::CUDAJLModel) = "CUDA.jl"
model_label(::JACCModel) = "JACC.jl"
model_label(::DaggerModel) = "Dagger.jl"

model_packages(::ExecutionModel) = String[]
model_packages(::CuNumericModel) = ["cuNumeric", "CUDA"]
model_packages(::CUDAJLModel) = ["CUDA"]
model_packages(::JACCModel) = ["JACC", "CUDA"]
model_packages(::DaggerModel) = ["Dagger", "CUDA"]

const MODEL_BY_ID = Dict(model_id(model) => model for model in EXECUTION_MODELS)

function execution_model(id::Symbol)
    return get(MODEL_BY_ID, id) do
        known = join(string.(sort!(collect(keys(MODEL_BY_ID)))), ", ")
        return error("Unknown execution model '$id'. Known models: $known")
    end
end

function parse_model(value)
    normalized = lowercase(replace(string(value), ".jl" => ""))
    aliases = Dict("cuda" => :cudajl, "cupy" => :cupynumeric)
    id = get(aliases, normalized, Symbol(normalized))
    execution_model(id) # validate before returning it
    return id
end

function parse_models(value)
    values = value isa AbstractVector ? value : [value]
    models = unique(parse_model.(values))
    isempty(models) && error("At least one execution model must be selected")
    return models
end

uses_fusion(::ExecutionModel) = false
uses_fusion(::CuNumericModel) = true

preparation_key(::ExecutionModel, run) = nothing
preparation_key(::CuNumericModel, run) = run.spec.fusion

prepare_model(::ExecutionModel, run, verbose; prepare_cunumeric) = nothing
function prepare_model(::CuNumericModel, run, verbose; prepare_cunumeric)
    return prepare_cunumeric(run.spec.fusion, verbose)
end

# Model-specific benchmark code is opt-in.  cuNumeric owns the accelerated
# variants; the other existing array baselines use only the non-accelerated
# definitions. JACC and Dagger have native Monte Carlo and GEMM workers.
supports_benchmark(::CuNumericModel, ::AbstractString) = true
function supports_benchmark(::Union{CuPyNumericModel,CUDAJLModel}, name::AbstractString)
    return !endswith(name, "_accelerated")
end
function supports_benchmark(::Union{JACCModel,DaggerModel}, name::AbstractString)
    return name in ("gemm", "montecarlo", "grayscott")
end

supports_gpu_count(::ExecutionModel, gpus::Integer) = gpus > 0
supports_gpu_count(::CUDAJLModel, gpus::Integer) = gpus == 1

function supports_run(model::ExecutionModel, name::AbstractString, gpus::Integer)
    supports_benchmark(model, name) || return false
    # JACC and Dagger run grayscott single-GPU only for now (JACC's 2D ghost path
    # is broken upstream; Dagger's @stencil multi-GPU still needs validation).
    model isa Union{JACCModel,DaggerModel} && startswith(name, "grayscott") && gpus != 1 &&
        return false
    return supports_gpu_count(model, gpus)
end

model_project(::CuNumericModel, root) = joinpath(root, "environments", "cunumeric")
model_project(::CUDAJLModel, root) = joinpath(root, "environments", "cuda")
model_project(::JACCModel, root) = joinpath(root, "environments", "jacc")
model_project(::DaggerModel, root) = joinpath(root, "environments", "dagger")

model_worker(::CuNumericModel, root) = joinpath(root, "src", "cunumeric", "single.jl")
model_worker(::CUDAJLModel, root) = joinpath(root, "src", "cuda", "single.jl")
model_worker(::CuPyNumericModel, root) = joinpath(root, "src", "cupynumeric", "single.py")
model_worker(::JACCModel, root) = joinpath(root, "src", "jacc", "single.jl")
model_worker(::DaggerModel, root) = joinpath(root, "src", "dagger", "single.jl")

struct WorkerRequest
    gpus::Int
    cpus::Int
    name::String
    T::String
    N::Int
    M::Int
    n_iter::Int
    n_warmup::Int
    n_trial::Int
    check_correctness::Bool
    n_correctness_iter::Int
    flops::Float64
end

function common_worker_args(r::WorkerRequest)
    return `$(r.gpus) $(r.name) $(r.T) $(r.N) $(r.M) $(r.n_iter) $(r.n_warmup) $(r.n_trial) $(r.check_correctness) $(r.n_correctness_iter) $(r.flops)`
end

function julia_worker_command(model::ExecutionModel, request::WorkerRequest, root)
    project = model_project(model, root)
    worker = model_worker(model, root)
    julia = get(ENV, "CUNUMERIC_BENCH_JULIA", joinpath(Sys.BINDIR, Base.julia_exename()))
    threads = max(request.cpus, 1)
    return `$julia --project=$project --threads=$threads $worker $(common_worker_args(request))`
end

function model_worker_command(
    model::Union{CuNumericModel,CUDAJLModel,JACCModel,DaggerModel}, request, root
)
    return julia_worker_command(model, request, root)
end

function model_worker_command(model::CuPyNumericModel, request, root)
    conda = get(ENV, "CUNUMERIC_BENCH_CONDA", get(ENV, "CONDA_EXE", "conda"))
    worker = model_worker(model, root)
    return `$conda run --no-capture-output -n $(cupynumeric_env_name()) python $worker $(common_worker_args(request))`
end

function model_environment(::ExecutionModel, request::WorkerRequest, verbose)
    return Dict{String,String}()
end

# JACC.Multi initializes every CUDA device visible to its process. Dagger can
# scope work to a subset, but using the same visibility rule keeps discovery,
# validation, and synchronization aligned with the planned GPU count. Preserve
# scheduler-provided identifiers (including UUIDs/MIG IDs) and select a prefix;
# use CUDA's logical integer identifiers only when no mask was supplied.
function selected_cuda_visibility(gpus::Integer; env=ENV)
    gpus > 0 || error("GPU count must be positive")
    visibility = get(env, "CUDA_VISIBLE_DEVICES", nothing)
    visibility === nothing && return join(0:(gpus - 1), ',')
    tokens = strip.(split(visibility, ','))
    any(isempty, tokens) && error("CUDA_VISIBLE_DEVICES contains an empty device identifier")
    length(unique(tokens)) == length(tokens) ||
        error("CUDA_VISIBLE_DEVICES contains duplicate device identifiers")
    length(tokens) >= gpus || error(
        "Requested $gpus GPUs, but CUDA_VISIBLE_DEVICES contains only $(length(tokens))"
    )
    return join(tokens[1:gpus], ',')
end

function model_environment(
    ::Union{JACCModel,DaggerModel}, request::WorkerRequest, verbose
)
    return Dict("CUDA_VISIBLE_DEVICES" => selected_cuda_visibility(request.gpus))
end

function model_environment(
    ::Union{CuNumericModel,CuPyNumericModel}, request::WorkerRequest, verbose
)
    config = "--cpus=$(request.cpus) --gpus=$(request.gpus)"
    if haskey(ENV, "CUNUMERIC_BENCH_FBMEM_MB")
        config *= " --fbmem=$(ENV["CUNUMERIC_BENCH_FBMEM_MB"])"
    end
    return Dict(
        "LEGATE_AUTO_CONFIG" => "1",
        "LEGATE_CONFIG" => config,
        "LEGATE_SHOW_CONFIG" => verbose ? "1" : "0",
    )
end

function wrapped_worker_command(model::ExecutionModel, request::WorkerRequest, root; verbose=false)
    runner = joinpath(root, "run_benchmark.sh")
    inner = model_worker_command(model, request, root)
    verbose_arg = verbose ? `--verbose` : ``
    return `bash $runner --model=$(model_id(model)) --gpus=$(request.gpus) --cpus=$(request.cpus) $verbose_arg -- $inner`
end

preflight_model(::ExecutionModel; kwargs...) = nothing

function preflight_model(
    ::CuPyNumericModel; env=ENV, which=Sys.which, check=success, root=nothing
)
    conda = get(env, "CUNUMERIC_BENCH_CONDA", get(env, "CONDA_EXE", "conda"))
    executable = which(conda)
    executable === nothing && error(
        "cuPyNumeric is enabled, but conda is not available to the worker. " *
        "Add conda to PATH or set CUNUMERIC_BENCH_CONDA to its executable path; " *
        "then run bash install_cupynumeric.sh. No benchmarks have been started.",
    )
    name = get(env, "CUPYNUMERIC_ENV", nothing)
    name === nothing && (name = cupynumeric_env_name())
    code = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('cupynumeric') else 1)"
    check(`$executable run --no-capture-output -n $name python -c $code`) || error(
        "Conda environment '$name' is unavailable or lacks cupynumeric. " *
        "Run bash install_cupynumeric.sh, or set CUPYNUMERIC_ENV to an existing environment. " *
        "No benchmarks have been started.",
    )
    return nothing
end

function preflight_julia_model(
    model::ExecutionModel, imports::String;
    env=ENV, which=Sys.which, check=success, root,
)
    project = model_project(model, root)
    isfile(joinpath(project, "Project.toml")) || error(
        "$(model_label(model)) is enabled, but its isolated environment is missing at $project"
    )
    julia = get(env, "CUNUMERIC_BENCH_JULIA", joinpath(Sys.BINDIR, Base.julia_exename()))
    executable = which(julia)
    executable === nothing && error("Julia executable '$julia' is unavailable")
    check(`$executable --project=$project -e $imports`) || error(
        "$(model_label(model)) is enabled, but its isolated environment is not instantiated. " *
        "Run `julia --project=$project -e 'using Pkg; Pkg.instantiate()'`. " *
        "No benchmarks have been started.",
    )
    return nothing
end

function preflight_model(model::JACCModel; kwargs...)
    return preflight_julia_model(model, "import JACC; JACC.@init_backend"; kwargs...)
end
function preflight_model(model::DaggerModel; kwargs...)
    return preflight_julia_model(model, "import Dagger; import CUDA"; kwargs...)
end
function preflight_model(model::CuNumericModel; kwargs...)
    return preflight_julia_model(model, "import cuNumeric"; kwargs...)
end
function preflight_model(model::CUDAJLModel; kwargs...)
    return preflight_julia_model(model, "import CUDA; import TensorOperations"; kwargs...)
end

function preflight_models(runs; kwargs...)
    isempty(runs) && return nothing
    root = normpath(joinpath(@__DIR__, ".."))
    seen = Set{Symbol}()
    for run in runs
        run.model in seen && continue
        push!(seen, run.model)
        preflight_model(execution_model(run.model); root, kwargs...)
    end
    return nothing
end

function isolated_model_versions(runs, root)
    result = Dict{String,Any}()
    for id in unique(run.model for run in runs)
        model = execution_model(id)
        packages = model_packages(model)
        isempty(packages) && continue
        manifest_path = joinpath(model_project(model, root), "Manifest.toml")
        isfile(manifest_path) || continue
        deps = get(TOML.parsefile(manifest_path), "deps", Dict{String,Any}())
        versions = Dict{String,String}()
        for package in packages
            entries = get(deps, package, Any[])
            isempty(entries) && continue
            entry = entries isa AbstractVector ? first(entries) : entries
            haskey(entry, "version") && (versions[package] = string(entry["version"]))
        end
        result[string(id)] = Dict(
            "project" => relpath(model_project(model, root), root),
            "versions" => versions,
        )
    end
    return result
end
