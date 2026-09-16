# Analytical estimates. This file is included after the benchmark definitions.
# All byte counts use BigInt so preflight cannot wrap on oversized dimensions.
Base.@kwdef struct MemoryContext
    model::Symbol = :cunumeric
    fusion::Bool = true
    gpus::Int = 1
    steps::Int = 1
    # Explicit upper bound per GPU for opaque native-library scratch/packing.
    workspace_bytes::Union{Nothing,Int} = nothing
end

struct MemoryEstimate
    initialization::BigInt
    iteration::BigInt
    workspace::BigInt
    explanation::String
end
peak_bytes(m::MemoryEstimate) = max(m.initialization, m.iteration) + m.workspace

# Per-GPU cuBLAS scratch; must match the deployed CUBLAS_WORKSPACE_CONFIG.
const CUBLAS_WORKSPACE_PER_GPU = 4 * 1024 * 1024
# Verified per-GPU native bounds by (benchmark, model); TOML overrides these.
const DEFAULT_WORKSPACE = Dict{Tuple{String,Symbol},Int}(
    ("gemm", :jacc) => 0, # custom parallel_for kernel, no cuBLAS
    ("gemm", :dagger) => CUBLAS_WORKSPACE_PER_GPU,
    ("gemm", :cudajl) => CUBLAS_WORKSPACE_PER_GPU,
    ("gemm", :cunumeric) => CUBLAS_WORKSPACE_PER_GPU,
    ("gemm", :cupynumeric) => CUBLAS_WORKSPACE_PER_GPU,
)

function library_workspace(b, c)
    bytes = c.workspace_bytes
    bytes === nothing && (bytes = get(DEFAULT_WORKSPACE, (name(b), c.model), nothing))
    bytes === nothing && error(
        "$(name(b)) / $(c.model): native workspace bound is unknown. " *
        "Set workspace_bytes to a verified per-GPU upper bound for this backend/library " *
        "configuration; autosizing will not guess or probe after an OOM.",
    )
    bytes >= 0 || error("workspace_bytes must be nonnegative")
    return big(bytes)
end

function validate_memory_context(b::AbstractBenchmark{T}, c) where {T}
    c.gpus > 0 || error("GPU count must be positive")
    c.steps > 0 || error("Trial steps must be positive")
    haskey(MODEL_BY_ID, c.model) || error("Unknown execution model $(c.model)")
    supports_gpu_count(execution_model(c.model), c.gpus) ||
        error("$(model_label(execution_model(c.model))) does not support $(c.gpus) GPUs")
    supports_benchmark(execution_model(c.model), name(b)) ||
        error("$(name(b)) is not implemented for $(model_label(execution_model(c.model)))")
    T in (Float32, Float64) ||
        error("Memory accounting currently supports Float32 and Float64; got $T")
    return all(>(0), dims(b)) || error("Problem dimensions must be positive")
end

# A conservative slab bound: rounding a partition up cannot undercount uneven
# dimensions. Stencil halos are counted separately below. DMD is never divided.
slab(n, tail, p) = cld(big(n), p) * big(tail)
function random_peak(elements, ::Type{T}, model) where {T}
    return elements * (model == :cupynumeric ? sizeof(Float64) + sizeof(T) : sizeof(T))
end

function memory_estimate(b::MonteCarloIntegration{T}, c::MemoryContext) where {T}
    validate_memory_context(b, c)
    e = cld(big(b.n_samples), c.gpus)
    bytes = e * sizeof(T)
    if c.model in (:cudajl, :jacc, :dagger)
        return MemoryEstimate(
            2bytes, 2bytes, 0,
            "partitioned samples plus conservative per-device reduction workspace; " *
            "model-native fused map/reduce",
        )
    end
    init = max(2bytes, random_peak(e, T, c.model))
    # The unfused NDArray copy path allocates an outer destination before
    # recursively materializing operations; count it as well as two temporaries.
    arrays = if c.model == :cupynumeric
        3
    else
        c.fusion ? 2 : 4
    end
    # Julia has tracing GC, not Python's reference counting. The returned
    # broadcast output is not explicitly destroyed by this baseline kernel.
    # This assumes the fully dotted expression in montecarlo.jl. On the unfused
    # path nested broadcast temporaries are explicitly destroyed by the runtime;
    # an undotted operation would instead escape that cleanup and need its own
    # per-iteration retention allowance.
    # Bound its retention over the complete trial instead of assuming a GC.
    retained = c.model == :cunumeric ? c.steps-1 : 0
    return MemoryEstimate(init, (arrays + retained)*bytes, 0,
        "samples + broadcast output; unfused temporaries; up to $retained prior Julia outputs awaiting GC; random dtype conversion",
    )
end

function memory_estimate(b::GEMM{T}, c::MemoryContext) where {T}
    validate_memory_context(b, c)
    # Without a mapper-specific replication guarantee count the complete inputs
    # on each GPU. This also covers broadcast operands in distributed matmul.
    a = big(b.N) * b.M * sizeof(T)
    out = big(b.N)^2 * sizeof(T)
    init = max(2a + out, a + random_peak(big(b.N)*b.M, T, c.model))
    return MemoryEstimate(init, 2a + out, library_workspace(b, c),
        "A, B, C; full operands per GPU (replication-safe); native packing/workspace")
end

function memory_estimate(b::AbstractGrayScott{T}, c::MemoryContext) where {T}
    validate_memory_context(b, c)
    b.N >= 3 && b.M >= 3 || error("Gray-Scott requires N and M >= 3")
    # Count full grids until the mapper's halo/replication contract is bounded.
    # Local expressions can retain parents; never treat a slice as a free copy.
    grid = big(b.N) * b.M * sizeof(T)
    interior = big(max(b.N-2, 0)) * max(b.M-2, 0) * sizeof(T)
    init = 4grid + random_peak(big(min(150, b.N, b.M))^2, T, c.model)
    # Four named RHS results plus an assignment output. Hard-scope acceleration
    # may eliminate these, but this remains a valid upper bound for every form.
    # Do not assume a lower peak solely from the @accelerate spelling.
    fused = c.model == :cudajl || (c.model == :cunumeric && c.fusion)
    # Unfused Laplacian: first branch survives evaluation of second branch;
    # include outer destination and intermediate binary operands.
    temps = fused ? 5 : 8
    variant = name(b)
    hard_scope =
        b isa Union{GrayScottAccelerated,GrayScottFunctionAccelerated,GrayScottLetAccelerated}
    # Hard scopes insert explicit last-use destruction whether fusion is on or
    # off. Baseline/begin/expression leave the named results for tracing GC.
    retained = c.model == :cunumeric && hard_scope || c.model == :cupynumeric ? 0 : 6*(c.steps-1)
    return MemoryEstimate(init, 4grid + (temps+retained)*interior, 0,
        "$variant: four persistent grids + $temps active interior buffers + $retained prior buffers awaiting GC; full-parent bound; fusion=$(c.fusion)",
    )
end

function memory_estimate(b::AbstractDMD{T}, c::MemoryContext) where {T}
    validate_memory_context(b, c)
    b.M >= 2 && b.N >= b.M-1 || error("DMD requires M >= 2 and N >= M-1")
    n, m, r = big(b.N), big(b.M-1), big(_dmd_rank(b))
    # Retained X, copies/views X1/X2, full thin U/Vt/S, projected products,
    # transpose copies, eigen inputs/outputs and complex lift. Count retained
    # parents even when only r columns are returned from _dmd_factors.
    persistent = n*big(b.M)
    factors = 3n*m + m*m + m
    project = 2m*r + 4n*r + 2r*r + r
    complex_lift = 2*(2n*r + 2r*r + r)
    # The factorization/output wrappers escape the lifetime rewriter; native
    # arrays can remain until GC between trials on Julia backends.
    retained_steps = c.model == :cupynumeric ? 1 : c.steps
    iteration = (persistent + retained_steps*(factors + project + complex_lift))*sizeof(T)
    init = random_peak(n*big(b.M), T, c.model)
    return MemoryEstimate(init, iteration, library_workspace(b, c),
        "full single-task SVD on one GPU (P does not divide memory); factors, retained parents, projections, complex lift",
    )
end

function memory_estimate(b::PoissonFFT{T}, c::MemoryContext) where {T}
    validate_memory_context(b, c)
    e = cld(big(b.M), c.gpus)*big(b.N)^2
    realbytes, complexbytes = e*sizeof(T), e*sizeof(Complex{T})
    kinv = big(b.N)^2*sizeof(T)
    # Python FFT precision is conservatively bounded by complex128.
    pycomplex = e*sizeof(ComplexF64)
    init = max(random_peak(e, T, c.model), realbytes+2complexbytes+kinv)
    iteration = c.model == :cupynumeric ? realbytes+2pycomplex+kinv : 2complexbytes+kinv
    return MemoryEstimate(init, iteration, library_workspace(b, c),
        "batched grids + replicated inverse Laplacian; Python out-of-place FFT precision; native FFT workspace",
    )
end

function memory_estimate(b::AbstractTensorContraction{T}, c::MemoryContext) where {T}
    validate_memory_context(b, c)
    # Full operands and intermediates, without assuming distributed packing.
    n = big(b.N)
    if b isa TensorProjection3
        live = (4n^3+n^2)*sizeof(T)
        init = max((2n^3+n^2)*sizeof(T), random_peak(n^3, T, c.model))
    else
        live = 3n^4*sizeof(T)
        init = max(live, n^4*sizeof(T)+random_peak(n^4, T, c.model))
    end
    return MemoryEstimate(init, live, library_workspace(b, c),
        "full contraction inputs/outputs and pairwise intermediates; native packing/workspace counted separately",
    )
end

function memory_estimate(b::AbstractConjugateGradient{T},c::MemoryContext) where {T}
    validate_memory_context(b,c)
    b.N>=2 && b.M==1 || error("CG requires N ≥ 2 and M=1")
    b.check_every>0 && b.max_iter>0 || error("CG check_every and max_iter must be positive")
    c.model==:jacc && b.N % c.gpus!=0 && error("JACC CG requires N divisible by GPUs")
    v=big(b.N)*sizeof(T)
    # Synchronization need not collect Julia wrappers for completed temporaries.
    retained = c.model == :cunumeric ? big(b.max_iter)*c.steps : 1
    return MemoryEstimate(16v,(24+24retained)*v,0,
        "CG bands/workspace and conservative retained iteration temporaries")
end
