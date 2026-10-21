module cuNumericKrylovExt

using cuNumeric: cuNumeric, NDArray, DeviceScalar, CNFloat, CNComplex, destroy!
using LinearAlgebra: axpy!, axpby!, rmul!
import Krylov

# Allocate through similar, preserving the NDArray storage type without needing
# S(undef, n). Dagger uses the same workspace-constructor integration point:
# https://github.com/JuliaParallel/Dagger.jl/blob/master/ext/KrylovExt.jl
for W in (:CgWorkspace, :BicgstabWorkspace, :GmresWorkspace, :MinresWorkspace,
          :LsqrWorkspace, :LsmrWorkspace)
    @eval function Krylov.$W(A, b::NDArray{T,1}; kwargs...) where {T}
        m, n = size(A)
        length(b) == m || throw(DimensionMismatch("right-hand side length must match the operator's row count"))
        vm, vn = similar(b, m), similar(b, n)
        try
            return Krylov.$W(Krylov.KrylovConstructor(vm, vn); kwargs...)
        finally
            destroy!(vm)
            destroy!(vn)
        end
    end
end

# Krylov's documented custom-vector hooks:
# https://jso.dev/Krylov.jl/stable/custom_workspaces/#Methods-to-overload-for-compatibility-with-Krylov.jl
# Its AbstractVector fallbacks operate on whole vectors (the n argument is
# unused), but require coefficients to match the vector element type. These
# methods keep runtime-backed coefficients in the existing cuNumeric operations.
function Krylov.kaxpy!(n::Integer, α::DeviceScalar, x::NDArray{T,1}, y::NDArray{T,1}) where {T}
    return axpy!(α, x, y)
end

function Krylov.kaxpby!(n::Integer, α::DeviceScalar, x::NDArray{T,1}, β::Number, y::NDArray{T,1}) where {T}
    return axpby!(α, x, β, y)
end

function Krylov.kaxpby!(n::Integer, α::Number, x::NDArray{T,1}, β::DeviceScalar, y::NDArray{T,1}) where {T}
    return axpby!(α, x, β, y)
end

function Krylov.kaxpby!(n::Integer, α::DeviceScalar, x::NDArray{T,1}, β::DeviceScalar, y::NDArray{T,1}) where {T}
    return axpby!(α, x, β, y)
end

Krylov.kscal!(n::Integer, α::DeviceScalar, x::NDArray{T,1}) where {T} = rmul!(x, α)
Krylov.kdiv!(n::Integer, x::NDArray{T,1}, α::DeviceScalar) where {T} = (x ./= α)
Krylov.kdivcopy!(n::Integer, y::NDArray{T,1}, x::NDArray{T,1}, α::DeviceScalar) where {T} = (y .= x ./ α)

const HostFloat = Union{Float32,Float64,ComplexF32,ComplexF64}
const DeviceFloat = Union{CNFloat,CNComplex}
host_coefficient(x::HostFloat) = x
function host_coefficient(x::DeviceFloat)
    cuNumeric._assert_allowfetch()
    return fetch(x)
end
host_givens(a, b) = Krylov.sym_givens(promote(host_coefficient(a), host_coefficient(b))...)

# Givens coefficients live in Krylov's host recurrence/workspace arrays.
Krylov.sym_givens(a::HostFloat, b::DeviceFloat) = host_givens(a, b)
Krylov.sym_givens(a::DeviceFloat, b::HostFloat) = host_givens(a, b)
Krylov.sym_givens(a::DeviceFloat, b::DeviceFloat) = host_givens(a, b)
Krylov.sym_givens(a::S, b::S) where {S<:CNFloat} = host_givens(a, b)

end
