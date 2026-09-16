using LinearAlgebra: Tridiagonal, norm

abstract type AbstractConjugateGradient{T} <: AbstractBenchmark{T} end

Base.@kwdef struct ConjugateGradientBenchmark{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end
Base.@kwdef struct ConjugateGradientAccelerated{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end
name(::ConjugateGradientAccelerated) = "cg"
name(::ConjugateGradientBenchmark) = "cg_plain"
dims(b::AbstractConjugateGradient) = (b.N, 1)
function data(b::AbstractConjugateGradient)
    return "CG: N=$(b.N), check_every=$(b.check_every), max_iter=$(b.max_iter)"
end
allowed_types(::Type{<:AbstractConjugateGradient}) = Union{Float32,Float64}
# Executed iterations depend on convergence; compare elapsed time, not nominal FLOPs.
total_flops(::AbstractConjugateGradient) = 0
estimate_scaling(b::AbstractConjugateGradient, p::Integer) = (scale_axis(b.N, p, 1), 1)
total_space(b::AbstractConjugateGradient{T}) where {T} = 7big(b.N)*sizeof(T)
correctness_uses_cpu(::AbstractConjugateGradient) = true

# Array-backend workers (cuNumeric, CUDA.jl) share this generic solver; cuNumeric
# adds an @accelerate specialization of cg_step! for ConjugateGradientAccelerated.
function initialize(b::AbstractConjugateGradient{T}; mod=cuNumeric) where {T}
    lower = mod.ones(T, b.N)
    diagonal = mod.ones(T, b.N)
    diagonal .*= T(4)
    upper = mod.ones(T, b.N)
    x = mod.zeros(T, b.N)
    r = mod.zeros(T, b.N)
    p = mod.zeros(T, b.N)
    Ap = mod.zeros(T, b.N)

    state = (; A=(lower, diagonal, upper), x, work=(r, p, Ap))
    return (state,)
end

# Generate the two variants from the same recurrence, as in Gray–Scott.
const CG_STEP_BODY = quote
    Ap .= diagonal .* p
    @views Ap[2:end] .+= lower[2:end] .* p[1:(end - 1)]
    @views Ap[1:(end - 1)] .+= upper[1:(end - 1)] .* p[2:end]
    # Zero residuals can occur before the next scheduled check.
    alpha = rho ./ max.(sum(p .* Ap), floatmin(T))
    x .+= alpha .* p
    r .-= alpha .* Ap
    next = sum(r .* r)
    p .= r .+ (next ./ max.(rho, floatmin(T))) .* p
    return next
end

# Plain step covers ConjugateGradientBenchmark everywhere, and the accelerated
# variant on backends without cuNumeric's @accelerate (e.g. the CUDA.jl worker).
let body = deepcopy(CG_STEP_BODY)
    @eval cg_step!(
        b::AbstractConjugateGradient{T}, x, r, p, Ap, lower, diagonal, upper, rho
    ) where {T} = $body
end

# Solve tridiag(1,4,1)*x = 1/2 from zero. All reductions stay in the DAG until checked.
function run!(b::AbstractConjugateGradient{T}, s) where {T}
    x = s.x
    r, p, Ap = s.work
    lower, diagonal, upper = s.A
    x .= zero(T)
    r .= T(0.5)
    p .= r
    rho = sum(r .* r)
    target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
    for k in 1:b.max_iter
        rho = cg_step!(b, x, r, p, Ap, lower, diagonal, upper, rho)
        if k % b.check_every == 0 || k == b.max_iter
            rr = only(rho)
            isfinite(rr) || error("CG produced a nonfinite residual")
            (rr <= target || b.max_iter == 1) && return k
        end
    end
    return error("CG did not converge within max_iter")
end

function check_benchmark_correctness(
    b::AbstractConjugateGradient{T}, gs::GlobalSettings; mod=cuNumeric
) where {T}
    n = min(b.N, 32)
    small = typeof(b)(; N=n, check_every=b.check_every, max_iter=b.max_iter)
    s = only(initialize(small; mod))
    run!(small, s)
    x = Array(s.x)
    A = Tridiagonal(ones(T, n-1), fill(T(4), n), ones(T, n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end

register_benchmark("cg", ConjugateGradientAccelerated)
register_benchmark("cg_plain", ConjugateGradientBenchmark)
