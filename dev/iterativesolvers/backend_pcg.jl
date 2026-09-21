# Investigation-only PCG with backend coefficients and a host stopping decision.
module BackendPCG
using cuNumeric, LinearAlgebra
import IterativeSolvers
import IterativeSolvers: done, converged

mutable struct PCGIterable{M,P,X,V,S,C}
    A::M
    Pl::P
    x::X
    r::V
    z::V
    p::V
    Ap::V
    residual::S
    tolerance::S
    rho::C
    alpha::C
    beta::C
    maxiter::Int
    iterations::Int
    checks::Int
    isconverged::Bool
end

function pcg_iterator!(x::NDArray{T,1}, A, b::NDArray{T,1};
                       Pl=IterativeSolvers.Identity(),
                       reltol::Real=sqrt(eps(real(T))), abstol::Real=zero(real(T)),
                       maxiter::Int=length(b), initially_zero::Bool=false) where {T}
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    reltol >= 0 && abstol >= 0 || throw(ArgumentError("tolerances must be nonnegative"))
    r, z, p, Ap = copy(b), similar(x), zero(x), similar(x)
    if !initially_zero
        mul!(Ap, A, x)
        r .-= Ap
    end
    residual = norm(r)
    R = real(T)
    tolerance = max.(R(reltol) .* residual, R(abstol))
    return PCGIterable(A, Pl, x, r, z, p, Ap, residual, tolerance,
                       NDArray(one(T)), NDArray(zero(T)), NDArray(zero(T)),
                       maxiter, 0, 0, false)
end

function done(it::PCGIterable, iteration::Int)
    # The only host extraction in the solver: a Boolean convergence decision.
    predicate = it.residual .<= it.tolerance
    it.isconverged = only(predicate)
    cuNumeric.destroy!(predicate)
    it.checks += 1
    return it.isconverged || iteration >= it.maxiter
end

converged(it::PCGIterable) = it.isconverged
Base.IteratorSize(::Type{<:PCGIterable}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:PCGIterable}) = Base.EltypeUnknown()

function Base.iterate(it::PCGIterable, iteration::Int=0)
    done(it, iteration) && return nothing
    ldiv!(it.z, it.Pl, it.r)
    rho = dot(it.z, it.r)
    it.beta .= rho ./ it.rho
    it.p .= it.z .+ it.beta .* it.p
    mul!(it.Ap, it.A, it.p)
    denominator = dot(it.p, it.Ap)
    it.alpha .= rho ./ denominator
    it.x .+= it.alpha .* it.p
    it.r .-= it.alpha .* it.Ap
    copyto!(it.rho, rho)
    cuNumeric.destroy!(rho)
    cuNumeric.destroy!(denominator)
    previous = it.residual
    it.residual = norm(it.r)
    cuNumeric.destroy!(previous)
    it.iterations = iteration + 1
    # Do not expose a residual that the next iteration destroys. Callers may
    # explicitly copy it.residual if they want a backend history.
    return it.iterations, it.iterations
end

function pcg!(x, A, b; kwargs...)
    it = pcg_iterator!(x, A, b; kwargs...)
    for _ in it
    end
    return it
end

pcg(A, b; kwargs...) = pcg!(zero(b), A, b; initially_zero=true, kwargs...)
end
