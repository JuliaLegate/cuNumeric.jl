using LinearAlgebra: Tridiagonal, norm

function initialize(b::ConjugateGradientBenchmark{T}; mod=cuNumeric) where {T}
    A = ntuple(_ -> mod.ones(T,b.N),3); A[2] .*= 4
    return ((; A, x=mod.zeros(T,b.N), work=ntuple(_ -> mod.zeros(T,b.N),3)),)
end

# Solve tridiag(1,4,1)*x = 1/2 from zero. All reductions stay in the DAG until checked.
function run!(b::ConjugateGradientBenchmark{T},s) where {T}
    x = s.x; r,p,Ap = s.work; lower,diagonal,upper = s.A
    x .= zero(T); r .= T(0.5); p .= r
    rho = sum(r .* r)
    target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
    for k in 1:b.max_iter
        Ap .= diagonal .* p
        @views Ap[2:end] .+= lower[2:end] .* p[1:end-1]
        @views Ap[1:end-1] .+= upper[1:end-1] .* p[2:end]
        # Zero residuals can occur before the next scheduled check.
        alpha = rho ./ max.(sum(p .* Ap),floatmin(T))
        x .+= alpha .* p
        r .-= alpha .* Ap
        next = sum(r .* r)
        p .= r .+ (next ./ max.(rho,floatmin(T))) .* p
        rho = next
        if k % b.check_every == 0 || k == b.max_iter
            rr = only(rho)
            isfinite(rr) || error("CG produced a nonfinite residual")
            (rr <= target || b.max_iter == 1) && return k
        end
    end
    error("CG did not converge within max_iter")
end

function check_benchmark_correctness(b::ConjugateGradientBenchmark{T},gs; mod=cuNumeric) where {T}
    n = min(b.N,32)
    small = ConjugateGradientBenchmark{T}(;N=n,check_every=b.check_every,max_iter=b.max_iter)
    s = only(initialize(small;mod)); run!(small,s); x = Array(s.x)
    A = Tridiagonal(ones(T,n-1),fill(T(4),n),ones(T,n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end
