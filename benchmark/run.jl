# run.jl: orchestrator, one child per benchmarks.toml entry. With args
# (<gpus> <name> <T> <N> <M> <iter> <warmup> <trial>) it runs one benchmark, e.g.
# `julia run.jl 1 grayscott Float32 1000 1000 100 5 5`
# Separate child per benchmark since LEGATE_CONFIG must be set before julia starts.

include("benchmarks.jl")
include("parse_benchmarks.jl")

function run_all_benchmarks(config="benchmarks.toml")
    gs, specs = parse_config(joinpath(@__DIR__, config))

    runner = joinpath(@__DIR__, "run_benchmark.sh")
    self = @__FILE__

    for spec in specs
        if !haskey(BENCHMARKS, spec.name)
            @warn "No benchmark registered for '$(spec.name)'; skipping."
            continue
        end

        N, M = spec.args
        println("\n================================")
        println(
            "$(spec.name): T=$(spec.T) gpus=$(spec.gpus) cpus=$(spec.cpus) N=$(N) M=$(M) " *
            "n_iter=$(gs.n_iter) n_warmup=$(gs.n_warmup) n_trial=$(gs.n_trial)",
        )
        println("================================")

        cmd = `bash $runner $self --gpus $(spec.gpus) --cpus $(spec.cpus) $(spec.name) $(spec.T) $N $M $(gs.n_iter) $(gs.n_warmup) $(gs.n_trial)`
        try
            run(cmd)
        catch e
            @error "Benchmark '$(spec.name)' failed; continuing." exception = e
        end
    end
end

# Resolve a TOML type string like "Float32" to the actual Julia type.
parse_type(s) = getfield(Base, Symbol(s))::DataType

function run_single(gpus, name, T_str, N, M, n_iter, n_warmup, n_trial)
    T = parse_type(T_str)
    b = BENCHMARKS[name]{T}(; N=N, M=M)
    gs = GlobalSettings(; n_warmup=n_warmup, n_iter=n_iter, n_trial=n_trial)

    println(
        "[cuNumeric] $(name) benchmark ($(T)) on $(N)x$(M) for $(n_iter) iterations " *
        "($(n_warmup) warmup) x $(n_trial) trials",
    )
    br = run_benchmark(b, gs)
    @printf("[cuNumeric] Mean Run Time: %.5f ± %.5f ms\n", mean(br.times_ms), _std(br.times_ms))
    @printf("[cuNumeric] FLOPS: %.5f ± %.5f GFLOPS\n", mean(br.gflops), _std(br.gflops))

    save_result(br, gpus)
end

if isempty(ARGS)
    run_all_benchmarks()
else
    using cuNumeric
    using LinearAlgebra
    gpus = parse(Int, ARGS[1])
    bench_name = ARGS[2]
    T_str = ARGS[3]
    N = parse(Int, ARGS[4])
    M = parse(Int, ARGS[5])
    n_iter = parse(Int, ARGS[6])
    n_warmup = parse(Int, ARGS[7])
    n_trial = parse(Int, ARGS[8])
    run_single(gpus, bench_name, T_str, N, M, n_iter, n_warmup, n_trial)
end
