# From benchmark/, run each case in a fresh process:
# bash run_benchmark.sh mwe_fusion_indexing.jl --gpus 8 --cpus 8 2454267008 fused
# bash run_benchmark.sh mwe_fusion_indexing.jl --gpus 8 --cpus 8 2454267072 fused
# bash run_benchmark.sh mwe_fusion_indexing.jl --gpus 8 --cpus 8 2454267072 native
# These straddle 2^31 at the last partition's START (7*N/8), not global N.
# With N near 2^31 and eight equal partitions, all starts still fit Int32.
# Also test the original failing N=4266645824, and repeat with --gpus 1.
# No RNG, sum, benchmark harness, or preference changes. Two persistent arrays.
using cuNumeric

function stage(f, label)
    println("START: $label")
    flush(stdout)
    result = f()
    cuNumeric.issue_execution_fence(; block=true)
    println("PASS: $label")
    flush(stdout)
    return result
end

function landmarks(n, p)
    points = Int[1,n]
    for boundary in (Int(2)^31, (cld(n,p)*k for k in 1:(p-1))...)
        append!(points,filter(i->1<=i<=n,(boundary-1,boundary,boundary+1)))
    end
    return sort!(unique!(points))
end

function main(args=ARGS)
    length(args)==3 || error("Use run_benchmark.sh with --gpus P --cpus C N fused|native")
    p,n = parse.(Int,args[1:2])
    mode = args[3]
    p>0 && n>0 || error("P and N must be positive")
    mode in ("fused","native") || error("Mode must be fused or native")
    println("P=$p N=$n mode=$mode; persistent data=$(2big(n)*sizeof(Float32)) bytes globally")
    println("Expected last partition start (zero-based, equal tiles): $((p-1)*cld(n,p)); Int32 limit=$(typemax(Int32))")
    x,y = stage("allocate and fill input/output") do
        cuNumeric.ones(Float32,n),cuNumeric.zeros(Float32,n)
    end
    points = landmarks(n,p)
    values = Dict(i=>Float32(0.125+0.025*j) for (j,i) in enumerate(points))
    stage("write boundary markers") do
        for i in points
            x[i:i] .= values[i]
        end
    end
    stage("verify input markers (before fusion)") do
        for i in points
            @assert only(Array(x[i:i])) == values[i] "input marker mismatch at $i"
        end
    end
    stage("$mode square / negate / exp") do
        # Exact broadcast tree for exp.(.-(x .^ 2)), including literal_pow.
        square = Base.Broadcast.broadcasted(Base.literal_pow,Ref(^),x,Ref(Val(2)))
        tree = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(exp,Base.Broadcast.broadcasted(-,square)))
        if mode == "fused"
            @assert cuNumeric.can_fuse_linear_broadcast(y,tree)
            cuNumeric.fuse_broadcast_tree!(y,tree)
        else
            # Force native tasks regardless of the user's fusion preference.
            tmp = cuNumeric.unravel_broadcast_tree(tree)
            copyto!(y,tmp)
            cuNumeric.destroy!(tmp)
        end
    end
    stage("verify output markers (no reduction)") do
        for i in points
            actual = only(Array(y[i:i]))
            expected = exp(-values[i]^2)
            @assert isapprox(actual,expected;rtol=2f-5,atol=2f-6) "output mismatch at $i: got $actual expected $expected"
        end
    end
    println("PASS: all $(length(points)) markers; P=$p N=$n mode=$mode")
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    main()
end
