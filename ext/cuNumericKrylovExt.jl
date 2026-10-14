module cuNumericKrylovExt

using cuNumeric: NDArray, DeviceScalar
using LinearAlgebra: axpy!, axpby!
import Krylov

# Allocate through similar, preserving the NDArray storage type without needing
# S(undef, n). Dagger uses the same workspace-constructor integration point:
# https://github.com/JuliaParallel/Dagger.jl/blob/master/ext/KrylovExt.jl
function Krylov.CgWorkspace(A, b::NDArray{T,1}) where {T}
    return Krylov.CgWorkspace(Krylov.KrylovConstructor(similar(b)))
end

function Krylov.BicgstabWorkspace(A, b::NDArray{T,1}) where {T}
    return Krylov.BicgstabWorkspace(Krylov.KrylovConstructor(similar(b)))
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

end
