#!/usr/bin/env julia

# Print @accelerate lifetime rewrites and the fused kernels launched by each
# Gray–Scott macro form. Run with a small grid, for example:
#   julia --project=benchmark benchmark/debug/grayscott_accelerate.jl 64

using cuNumeric

const BENCHMARK_SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(BENCHMARK_SRC, "core.jl"))
include(joinpath(BENCHMARK_SRC, "benchmarks", "grayscott.jl"))
include(joinpath(BENCHMARK_SRC, "benchmarks", "grayscott_accelerate_forms.jl"))

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
N >= 4 || error("N must be at least 4")

const FORMS = (
    (:function, "function", GrayScottFunctionAccelerated),
    (:begin, "begin", GrayScottBeginAccelerated),
    (:let, "let", GrayScottLetAccelerated),
    (:expression, "expression", GrayScottExpressionAccelerated),
)

function lifetime_expansion(kind)
    body = deepcopy(GRAYSCOTT_STEP_BODY)
    input = if kind === :function
        Expr(:function, Expr(:call, :debug_step, :u, :v, :u_new, :v_new, :args), body)
    elseif kind === :let
        Expr(:let, body)
    else
        body
    end
    return cuNumeric._accelerate_expand(input, @__MODULE__)
end

function print_lifetimes(kind, label)
    println("\n", "="^80, "\n", uppercase(label), " — lifetime analysis\n", "="^80)
    if kind === :expression
        # Expression form accelerates each assignment independently.
        for statement in GRAYSCOTT_STEP_BODY.args
            statement isa LineNumberNode && continue
            statement isa Expr && statement.head === :(=) || continue
            lhs, rhs = statement.args
            println("\nRHS: ", lhs)
            expansion = cuNumeric._accelerate_expand(rhs, @__MODULE__)
            cuNumeric.print_lifetime_analysis(expansion)
        end
    else
        cuNumeric.print_lifetime_analysis(lifetime_expansion(kind))
    end
end

function run_form(label, type)
    println("\n", "="^80, "\n", uppercase(label), " — runtime kernels\n", "="^80)
    b = build_benchmark(type, Float32, N, N)
    state = only(initialize(b; deterministic=true))
    return run!(b, state)
end

println("Gray–Scott @accelerate debug; grid=$(N)x$(N)")
cuNumeric.BCAST_FUSION_DEBUG[] = true
println("BCAST_FUSION_DEBUG = ", cuNumeric.BCAST_FUSION_DEBUG[])
for (kind, label, type) in FORMS
    print_lifetimes(kind, label)
    run_form(label, type)
end
