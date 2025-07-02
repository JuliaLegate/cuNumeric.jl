using Statistics

import Random

function loop_array(iters::Integer, size::Integer = 500_000)
    rng = Random.MersenneTwister()
    t = @timed begin
        for _ in range(1,iters)
            arr = rand(rng, size)
            arr[1] = 1
            GC.gc(false)
        end
    end
    return t
end

N_trials = 40
N_iters_per_trial = 200

res = loop_array(N_iters_per_trial)
GC.gc(true)

non_gc_time_ms = Float64[]
bytes_allocated = Float64[]
gc_times_ms = Float64[]
pool_allocs = Float64[]
non_pool_gc_allocs = Float64[]
malloc_calls = Float64[]
realloc_calls = Float64[]
minor_collections = Float64[]
full_collections = Float64[]
for _ in range(1,N_trials)
    res = loop_array(N_iters_per_trial)
    push!(bytes_allocated, res.gcstats.allocd)
    push!(non_gc_time_ms, (res.time - res.gctime )* 1e3 )
    push!(gc_times_ms, res.gcstats.total_time / 1e6)
    push!(pool_allocs, res.gcstats.poolalloc)
    push!(non_pool_gc_allocs, res.gcstats.bigalloc)
    push!(malloc_calls, res.gcstats.malloc)
    push!(realloc_calls, res.gcstats.realloc)
    minor_collects = res.gcstats.pause - res.gcstats.full_sweep
    push!(minor_collections, minor_collects)
    push!(full_collections, res.gcstats.full_sweep)
    GC.gc(true)
end

iter_per_collection = N_iters_per_trial ./ (minor_collections .+ full_collections)
gc_time_per_collection_ms = gc_times_ms ./ (minor_collections .+ full_collections)
mean(non_gc_time_ms)