using CppGcTest

@noinline function loop_array_no_time(iters::Integer, size::Integer = 500_000)
    for _ in range(1,iters)
        arr = CppGcTest.CppVector(size)
        CppGcTest.set(arr, 1, 1.0)
        GC.gc(false)
    end
end

@time loop_array_no_time(1000)
