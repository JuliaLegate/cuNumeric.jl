# State only: Base.iterate is supplied by IterativeSolvers.
mutable struct NDArrayCGIterable{M,X,V,S} <: IS.AbstractCGIterable
    A::M
    x::X
    r::V
    c::V
    u::V
    tol::S
    residual::S
    prev_residual::S
    maxiter::Int
    mv_products::Int
    checked_residual::Union{Nothing,S}
    isconverged::Bool
end

mutable struct NDArrayPCGIterable{P,M,X,V,S,C} <: IS.AbstractPCGIterable
    Pl::P
    A::M
    x::X
    r::V
    c::V
    u::V
    tol::S
    residual::S
    ρ::C
    maxiter::Int
    mv_products::Int
    checked_residual::Union{Nothing,S}
    isconverged::Bool
end

const NDArrayCG = Union{NDArrayCGIterable,NDArrayPCGIterable}

function IS.converged(it::NDArrayCG)
    if it.checked_residual !== it.residual
        predicate = it.residual .<= it.tol
        it.isconverged = only(predicate)
        cuNumeric.destroy!(predicate)
        it.checked_residual = it.residual
    end
    return it.isconverged
end

IS.done(it::NDArrayCG, iteration::Int) = IS.converged(it) || iteration >= it.maxiter
IS.cg_history_type(it::NDArrayCG) = typeof(it.residual)
IS.cg_check_verbose(::NDArrayCG) = throw(ArgumentError(
    "verbose=true requires scalar residual display; use log=true and explicitly " *
    "extract history[:resnorm] after the solve to keep synchronization at convergence checks."))

function IS.cg_iterator!(x::NDArray{T,1}, A, b::NDArray{T,1}, Pl=IS.Identity();
                        abstol::Real=zero(real(T)), reltol::Real=sqrt(eps(real(T))),
                        maxiter::Int=size(A, 2),
                        statevars::IS.CGStateVariables=IS.CGStateVariables(zero(x), similar(x), similar(x)),
                        initially_zero::Bool=false) where {T}
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    abstol >= 0 && reltol >= 0 || throw(ArgumentError("tolerances must be nonnegative"))
    u, r, c = statevars.u, statevars.r, statevars.c
    fill!(u, zero(T))
    copyto!(r, b)
    mv_products = 0
    if !initially_zero
        mul!(c, A, x)
        r .-= c
        mv_products = 1
    end
    residual = norm(r)
    R = real(T)
    tolerance = max.(R(reltol) .* residual, R(abstol))
    return _cg_state(Pl, A, x, r, c, u, tolerance, residual, maxiter, mv_products)
end

function _cg_state(::IS.Identity, A, x, r, c, u, tol, residual, maxiter, mv_products)
    return NDArrayCGIterable(A, x, r, c, u, tol, residual,
                            NDArray(one(eltype(residual))), maxiter, mv_products, nothing, false)
end

function _cg_state(Pl, A, x, r, c, u, tol, residual, maxiter, mv_products)
    return NDArrayPCGIterable(Pl, A, x, r, c, u, tol, residual,
                             NDArray(one(eltype(x))), maxiter, mv_products, nothing, false)
end
