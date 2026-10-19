# Call rules used ONLY during @accelerate expansion. Positions refer to the
# positional arguments of a call, not to runtime objects or their fields.
#
# Independent()         : an independent result, cleaned up after its last use.
# DependsOn((1, ...))   : a borrowed result depending on selected arguments.
# DependsOnAll()        : a borrowed result depending on all positional arguments.
# UnknownDependencies() : keep the result and arguments until scope exit.
#
# Unknown functions are assumed to return locally owned results and not retain
# arguments through side effects. Functions returning caller-owned aliases need
# a borrowing rule. If an unknown result escapes, preserve its arguments too.
# Rules describe the callable's supported numerical methods; they do not inspect
# function bodies. Define extension rules before expanding the accelerated code.
struct Independent end
struct DependsOnAll end
struct UnknownDependencies end

# Positions are data, not type parameters. N fixes the tuple's length so its
# field stays concrete without specializing on each combination of positions.
struct DependsOn{N}
    positions::NTuple{N,Int}
end

_lifetime_dependencies(::Any) = UnknownDependencies()
_lifetime_dependencies(f, ::Val) = _lifetime_dependencies(f)

# These loops define ordinary dispatch methods once, when the module loads.
for f in (Base.transpose, Base.adjoint, identity, parent, getproperty, getfield,
    first, last, cnscalar, copyto!, fill!, setindex!, LinearAlgebra.mul!)
    @eval _lifetime_dependencies(::typeof($f)) = DependsOn((1,))
end

for T in (CNScalar, LinearAlgebra.Transpose, LinearAlgebra.Adjoint,
    LinearAlgebra.Diagonal, LinearAlgebra.Cholesky,
    LinearAlgebra.LowerTriangular, LinearAlgebra.UpperTriangular)
    @eval _lifetime_dependencies(::Type{<:$T}) = DependsOn((1,))
end
_lifetime_dependencies(::Type{<:NDArrayQR}) = DependsOn((1, 2))
_lifetime_dependencies(::Type{<:LinearAlgebra.SVD}) = DependsOn((1, 2, 3))
_lifetime_dependencies(::Type{<:LinearAlgebra.Eigen}) = DependsOn((1, 2))

_lifetime_dependencies(::Type{<:Array}) = Independent()
# Unlike Base.transpose, cuNumeric.transpose creates an independent store handle.
for f in (+, -, *, /, \, ^, sin, cos, tan, exp, log, sqrt, abs, abs2,
    zeros, ones, similar, copy, transpose)
    @eval _lifetime_dependencies(::typeof($f)) = Independent()
end
# Unary arithmetic may return the input itself. Keep its owner even when a
# particular implementation happens to copy; binary numerical results are owned.
for f in (+, *)
    @eval _lifetime_dependencies(::typeof($f), ::Val{1}) = DependsOn((1,))
end
