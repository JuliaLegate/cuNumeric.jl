using LinearAlgebra: Tridiagonal, norm

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
let body = quote
        Ap .= diagonal .* p
        @views Ap[2:end] .+= lower[2:end] .* p[1:end-1]
        @views Ap[1:end-1] .+= upper[1:end-1] .* p[2:end]
        # Zero residuals can occur before the next scheduled check.
        alpha = rho ./ max.(sum(p .* Ap),floatmin(T))
        x .+= alpha .* p
        r .-= alpha .* Ap
        next = sum(r .* r)
        p .= r .+ (next ./ max.(rho,floatmin(T))) .* p
        return next
    end
    @eval cg_step!(b::AbstractConjugateGradient{T},x,r,p,Ap,lower,diagonal,upper,rho) where {T} = $body
    if CUNUMERIC_BENCH_RUNTIME
        signature = :(cg_step!(b::ConjugateGradientAccelerated{T},x,r,p,Ap,lower,diagonal,upper,rho) where {T})
        @eval $(_define_accelerated_definition(signature,body))
    end
end

# Solve tridiag(1,4,1)*x = 1/2 from zero. All reductions stay in the DAG until checked.
function run!(b::AbstractConjugateGradient{T},s) where {T}
    x = s.x; r,p,Ap = s.work; lower,diagonal,upper = s.A
    x .= zero(T); r .= T(0.5); p .= r
    rho = sum(r .* r)
    target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
    for k in 1:b.max_iter
        rho = cg_step!(b,x,r,p,Ap,lower,diagonal,upper,rho)
        if k % b.check_every == 0 || k == b.max_iter
            rr = only(rho)
            isfinite(rr) || error("CG produced a nonfinite residual")
            (rr <= target || b.max_iter == 1) && return k
        end
    end
    error("CG did not converge within max_iter")
end

function check_benchmark_correctness(b::AbstractConjugateGradient{T},gs; mod=cuNumeric) where {T}
    n = min(b.N,32)
    small = typeof(b)(;N=n,check_every=b.check_every,max_iter=b.max_iter)
    s = only(initialize(small;mod)); run!(small,s); x = Array(s.x)
    A = Tridiagonal(ones(T,n-1),fill(T(4),n),ones(T,n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end
