# Internal @accelerate extension point: return a tuple of values whose handles
# this value borrows. Dependencies must be acyclic and unchanged while a binding
# is live. Optional-package methods belong in their package extensions.
_lifetime_dependencies(::Any) = ()
_lifetime_dependencies(x::Tuple) = x
_lifetime_dependencies(x::NamedTuple) = values(x)

_lifetime_dependencies(x::CNScalar) = (x.value,)
_lifetime_dependencies(x::Union{Transpose,Adjoint}) = (parent(x),)
_lifetime_dependencies(x::Diagonal) = (x.diag,)
_lifetime_dependencies(x::NDArrayQR) = (x.Q, x.R)
_lifetime_dependencies(x::SVD) = (x.U, x.S, x.Vt)
_lifetime_dependencies(x::Eigen) = (x.values, x.vectors)
_lifetime_dependencies(x::Cholesky) = (x.factors,)
# Cholesky exposes its stored factor through .L, .U, and .UL wrappers.
_lifetime_dependencies(x::Union{LowerTriangular,UpperTriangular}) = (parent(x),)
