# RAM query + binary search. Peak bytes and P-scaling live on each benchmark
# type (`total_space`, `estimate_scaling`, `fit_one_gpu`) so the formulas sit
# next to the kernel. Python never sees them: run.jl resolves N/M/flops and
# passes integers on the worker command line.

align8(n::Integer) = max(8, 8 * fld(Int(n), 8))
align2(n::Integer) = max(2, 2 * fld(Int(n), 2))

function scale_axis(n1::Integer, P::Integer, α::Real)
    return align8(floor(Int, n1 * Float64(P)^Float64(α)))
end

is_auto_size(x) = x isa AbstractString && lowercase(strip(string(x))) == "auto"

function largest_feasible(lo::Int, hi::Int, pred)
    hi < lo && return nothing
    pred(lo) || return nothing
    best = lo
    while lo <= hi
        mid = lo + (hi - lo) >> 1
        if pred(mid)
            best = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return best
end

function min_gpu_memory_bytes()
    out = read(`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, String)
    mibs = Int[]
    for line in split(out, '\n'; keepempty=false)
        tok = strip(split(line, ',')[1])
        push!(mibs, parse(Int, tok))
    end
    isempty(mibs) && error("nvidia-smi reported no GPUs")
    return minimum(mibs) * 1024^2
end

function autosize_budget(mem_frac::Real)
    frac = parse(Float64, get(ENV, "CUNUMERIC_BENCH_MEM_FRAC", string(mem_frac)))
    (0 < frac <= 1) || error("mem_frac must be in (0, 1], got $frac")
    bytes = Int(floor(frac * min_gpu_memory_bytes()))
    bytes > 0 || error("GPU memory budget is 0")
    return bytes, frac
end

function parse_bench_type(T_str::AbstractString)
    return getfield(Base, Symbol(T_str))::DataType
end

function resolve_autosize(
    name::AbstractString, T_str::AbstractString, P::Integer;
    mem_frac::Real, N_hint, M_hint,
)
    haskey(BENCHMARKS, name) || error(
        "No benchmark registered for '$(name)'. Known: $(join(sort(collect(keys(BENCHMARKS))), ", "))",
    )
    B = BENCHMARKS[name]
    T = parse_bench_type(T_str)
    budget, frac = autosize_budget(mem_frac)
    N1, M1 = fit_one_gpu(B, T; budget, N_hint, M_hint)
    scaled = estimate_scaling(build_benchmark(B, T, N1, M1), P)
    if scaled === nothing
        return nothing
    end
    N, M = scaled
    return (; N, M, N1, M1, budget, frac)
end
