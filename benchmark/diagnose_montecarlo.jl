# From benchmark/: bash run_benchmark.sh --model=cunumeric --gpus=8 --cpus=8 -- julia --project=. diagnose_montecarlo.jl 8 4266645824
# Diagnostic only: fences deliberately change execution timing.
using cuNumeric

length(ARGS) == 2 || error("Pass <P> <N>; see the run_benchmark.sh example above")
const N = parse(Int, ARGS[2])
N > 0 || error("N must be positive")
println("Monte Carlo diagnostic: GPUs=$(ARGS[1]), N=$N, fusion=$(cuNumeric.FUSE_BROADCAST_EXPRS)")

function stage(f, label)
    println("START: $label")
    flush(stdout)
    result = f()
    cuNumeric.issue_execution_fence(; block=true)
    println("PASS: $label")
    flush(stdout)
    return result
end

x = stage("Float32 random generation") do
    return cuNumeric.rand(Float32, N)
end
x = stage("scale samples") do
    return 10.0f0 .* x
end
y = stage("fused square / negate / exponential") do
    return exp.(.-(x .^ 2))
end
s = stage("sum reduction") do
    return sum(y)
end
stage("scale reduction result") do
    return (10.0f0 / N) * s
end
