using InteractiveUtils

@noinline function loop_array_no_time(iters::Integer, size::Integer = 500_000)
    for _ in range(1,iters)
        arr = zeros(size)
        arr[1] = 1
        GC.gc(false)
    end
end

@code_warntype loop_array_no_time(1000)
