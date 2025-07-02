using cuNumeric
using BenchmarkTools

@noinline function loop_array_no_time(iters::Integer, size::Integer=500_000)
    for _ in range(1, iters)
        arr = cuNumeric.full(Tuple(Int64(size)), 3.1415)
        GC.gc(false)
    end
end

@btest loop_array_no_time(1000)
