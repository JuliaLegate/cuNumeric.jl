# One sample per fresh Julia process; see README.md in this directory.
# Construct syntax before timing so parsing does not count as macro expansion.
mode = get(ARGS, 1, "function")
if mode == "precompile"
    stats = @timed Base.compilecache(Base.identify_package("cuNumeric"))
    println("LATENCY,precompile,", stats.time)
    exit()
end

inputs = Dict(
    "function" => :(@accelerate function latency_op(a, b, dest)
        product = @. a * b
        shifted = product .+ 0.25f0
        dest .= shifted .* 2f0
        return dest
    end),
    "begin" => :(function latency_op(a, b, dest)
        @accelerate begin
            product = a .* b
            shifted = product .+ 0.25f0
            dest .= shifted .* 2f0
        end
        return dest
    end),
    "multi" => :(function latency_op(a, b, dest)
        @accelerate begin
            product = a .* b
            shifted = product .+ 0.25f0
            (product, shifted)
        end
    end),
    "let" => :(function latency_op(a, b, dest)
        @accelerate let
            product = a .* b
            shifted = product .+ 0.25f0
            dest .= shifted .* 2f0
            dest
        end
    end),
    "expression" => :(function latency_op(a, b, dest)
        @accelerate dest .= (a .* b .+ 0.25f0) .* 2f0
    end),
    "stencil" => :(@accelerate function latency_op(a, b, dest)
        lap = a[3:end, 2:(end - 1)] .- 2f0 .* a[2:(end - 1), 2:(end - 1)] .+
              a[1:(end - 2), 2:(end - 1)]
        reaction = a[2:(end - 1), 2:(end - 1)] .* b[2:(end - 1), 2:(end - 1)]
        dest[2:(end - 1), 2:(end - 1)] = lap .+ reaction
        return dest
    end),
    "plain" => :(function latency_op(a, b, dest)
        dest .= (a .* b .+ 0.25f0) .* 2f0
    end),
)

load_stats = @timed @eval using cuNumeric
println("LATENCY,load,", load_stats.time)
println("LATENCY,load_compile,", get(load_stats, :compile_time, NaN))
println("CONFIG,julia,", VERSION)
println("CONFIG,fusion,", cuNumeric.FUSE_BROADCAST_EXPRS)
println("CONFIG,gpu,", cuNumeric._has_gpu_target())
mode == "load" && exit()
input = inputs[mode]

expansion_stats = @timed Base.macroexpand(@__MODULE__, deepcopy(input))
println("LATENCY,expand_first,", expansion_stats.time)
println("LATENCY,expand_compile,", get(expansion_stats, :compile_time, NaN))
println("BYTES,expand_first,", expansion_stats.bytes)
warm_expansion = minimum(
    @elapsed(Base.macroexpand(@__MODULE__, deepcopy(input))) for _ in 1:5
)
println("LATENCY,expand_warm,", warm_expansion)
definition_stats = @timed Core.eval(@__MODULE__, expansion_stats.value)
println("LATENCY,definition,", definition_stats.time)
get(ENV, "LATENCY_EXPANSION_ONLY", "false") == "true" && exit()

# Data construction and its compilation are reported separately from the first
# operation. The execution fence makes GPU timings include completed work.
host_a = reshape(Float32.(1:1024) ./ 1024f0, 32, 32)
host_b = reverse(host_a; dims=1)
allocation_stats = @timed begin
    a = NDArray(host_a)
    b = NDArray(host_b)
    dest = NDArray(zeros(Float32, size(host_a)))
    cuNumeric.issue_execution_fence(; block=true)
end
println("LATENCY,inputs,", allocation_stats.time)

function measured_call(a, b, dest)
    result = Base.invokelatest(latency_op, a, b, dest)
    cuNumeric.issue_execution_fence(; block=true)
    return result
end
first_stats = @timed measured_call(a, b, dest)
println("LATENCY,execute_first,", first_stats.time)
println("LATENCY,execute_compile,", get(first_stats, :compile_time, NaN))
println("BYTES,execute_first,", first_stats.bytes)
println("LATENCY,first_use,", expansion_stats.time + definition_stats.time + first_stats.time)

# Validate before reporting steady-state time. No correctness work is timed.
cuNumeric.@allowscalar if mode == "multi"
    product, shifted = first_stats.value
    @assert Array(product) ≈ host_a .* host_b
    @assert Array(shifted) ≈ host_a .* host_b .+ 0.25f0
    foreach(cuNumeric.destroy!, first_stats.value)
elseif mode == "stencil"
    expected = zeros(Float32, size(host_a))
    expected[2:(end - 1), 2:(end - 1)] .=
        host_a[3:end, 2:(end - 1)] .- 2f0 .* host_a[2:(end - 1), 2:(end - 1)] .+
        host_a[1:(end - 2), 2:(end - 1)] .+
        host_a[2:(end - 1), 2:(end - 1)] .* host_b[2:(end - 1), 2:(end - 1)]
    @assert isapprox(Array(dest), expected; rtol=1f-5, atol=1f-6)
else
    @assert Array(dest) ≈ (host_a .* host_b .+ 0.25f0) .* 2f0
end

warm = Float64[]
for _ in 1:5
    sample = @timed measured_call(a, b, dest)
    push!(warm, sample.time)
    mode == "multi" && foreach(cuNumeric.destroy!, sample.value)
end
println("LATENCY,execute_warm,", minimum(warm))
foreach(cuNumeric.destroy!, (a, b, dest))
println("VALIDATED,", mode)
