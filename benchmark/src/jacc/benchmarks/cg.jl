using LinearAlgebra: Tridiagonal, norm

Base.@kwdef struct JACCCG{T}
    N::Int
    gpus::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end
function model_build_cg(c::ModelWorkerConfig)
    c.M==1 && c.N % c.gpus==0 || error("JACC CG requires M=1 and N divisible by GPUs")
    return JACCCG{c.T}(;c.kwargs...,N=c.N,gpus=c.gpus)
end
function model_initialize(b::JACCCG{T}) where {T}
    JACC.Multi.ndev()==b.gpus || error("JACC visible device count differs from requested GPUs")
    A = (JACC.Multi.array(ones(T,b.N)), JACC.Multi.array(fill(T(4),b.N)), JACC.Multi.array(ones(T,b.N)))
    return (; A, x=JACC.Multi.array(zeros(T,b.N)),
        work=ntuple(i -> JACC.Multi.array(zeros(T,b.N);ghost_dims=(i==2 ? 1 : 0)),3))
end
model_synchronize(::JACCCG) = nothing # Multi operations synchronize all devices.

# Tridiagonal/AXPY kernels adapted from JACC-Test-Codes,
# 2231bb261c4c8fbb45e061327646352c9844a628, src/JACCTestCodes.jl.
function cg_matvec(i,lower,diagonal,upper,p,Ap)
    j = JACC.Multi.ghost_shift(i,p)
    @inbounds Ap[i] = diagonal[i]*p[j] + (j>1 ? lower[i]*p[j-1] : zero(diagonal[i])) +
        (j<length(p) ? upper[i]*p[j+1] : zero(diagonal[i]))
end
cg_product(i,x,y) = @inbounds x[JACC.Multi.ghost_shift(i,x)]*y[JACC.Multi.ghost_shift(i,y)]
cg_axpy(i,x,a,p) = (@inbounds x[i] += a*p[JACC.Multi.ghost_shift(i,p)])
function cg_direction(i,p,r,beta)
    j = JACC.Multi.ghost_shift(i,p)
    @inbounds p[j] = r[i]+beta*p[j]
end
function cg_reset(i,x,r,p,value)
    @inbounds x[i] = zero(value)
    @inbounds r[i] = p[JACC.Multi.ghost_shift(i,p)] = value
end

function model_run!(b::JACCCG{T},s) where {T}
    x = s.x; r,p,Ap = s.work
    dot(x,y) = only(JACC.Multi.parallel_reduce(b.N,cg_product,x,y))
    JACC.Multi.parallel_for(b.N,cg_reset,x,r,p,T(0.5))
    rho = dot(r,r)
    target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
    for k in 1:b.max_iter
        JACC.Multi.sync_ghost_elems!(p)
        JACC.Multi.parallel_for(b.N,cg_matvec,s.A...,p,Ap)
        alpha = rho/max(dot(p,Ap),floatmin(T))
        JACC.Multi.parallel_for(b.N,cg_axpy,x,alpha,p)
        JACC.Multi.parallel_for(b.N,cg_axpy,r,-alpha,Ap)
        next = dot(r,r)
        JACC.Multi.parallel_for(b.N,cg_direction,p,r,next/max(rho,floatmin(T)))
        rho = next
        if k % b.check_every == 0 || k == b.max_iter
            isfinite(rho) || error("CG produced a nonfinite residual")
            (rho <= target || b.max_iter == 1) && return k
        end
    end
    error("CG did not converge within max_iter")
end

function model_check_correctness(b::JACCCG{T},config) where {T}
    n = min(b.N,max(b.gpus,fld(32,b.gpus)*b.gpus))
    small = JACCCG{T}(;N=n,gpus=b.gpus,check_every=b.check_every,max_iter=b.max_iter)
    s = model_initialize(small); model_run!(small,s); x = JACC.to_host(s.x)
    A = Tridiagonal(ones(T,n-1),fill(T(4),n),ones(T,n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end
