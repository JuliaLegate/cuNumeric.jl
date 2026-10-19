# Run from the package environment. This measures expansion, not GPU launches.
using cuNumeric

chain = :(function chain(a, b)
    x = a .* b
    y = x .+ 1
    z = y .* 2
    z .+ 3
end)
source = Meta.parseall(read(joinpath(@__DIR__, "..", "test", "workflows", "grayscott.jl"), String))
grayscott = first(
    filter(source.args) do expression
        return expression isa Expr && expression.head === :macrocall && first(expression.args) === Symbol(
            "@accelerate"
        )
    end,
).args[end]

println("Julia ", VERSION, ", fusion=", cuNumeric.FUSE_BROADCAST_EXPRS)
for (name, expression) in (("chain", chain), ("grayscott", grayscott))
    expand() = cuNumeric._accelerate_expand(expression, Main)
    foreach(_ -> expand(), 1:5)
    samples = [@timed(expand()) for _ in 1:30]
    println(name, ": ", 1000minimum(s.time for s in samples), " ms, ",
        minimum(s.bytes for s in samples), " bytes")
end

@accelerate function scalar_cleanup(x::Float64)
    y = x + 1.0
    y * 2.0
end
cleanup_bytes(x) = @allocated scalar_cleanup(x)
cleanup_bytes(3.0)
println("scalar cleanup execution: ", cleanup_bytes(3.0), " bytes")
