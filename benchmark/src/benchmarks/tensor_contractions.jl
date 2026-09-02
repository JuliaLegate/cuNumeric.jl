using TensorOperations

abstract type AbstractTensorContraction{T} <: AbstractBenchmark{T} end

Base.@kwdef struct TensorProjection3{T} <: AbstractTensorContraction{T}
    N::Int
end

Base.@kwdef struct TensorContract4{T} <: AbstractTensorContraction{T}
    N::Int
end

name(::TensorProjection3) = "tensor_projection3"
name(::TensorContract4) = "tensor_contract4"
dims(b::AbstractTensorContraction) = (b.N, 1)
function data(b::AbstractTensorContraction{T}) where {T}
    return "$(name(b)) with T=$(T), N=$(b.N)"
end

allowed_types(::Type{<:AbstractTensorContraction}) = cuNumeric.SUPPORTED_FLOAT_TYPES

# Three rank-3 outputs, each containing an N-term dot product.
total_flops(b::TensorProjection3) = 3 * b.N^3 * (2 * b.N - 1)

# N^4 output elements, each containing an N^2-term dot product.
total_flops(b::TensorContract4) = b.N^4 * (2 * b.N^2 - 1)

function build_benchmark(
    ::Type{TensorProjection3}, ::Type{T}, N, M
) where {T}
    return TensorProjection3{T}(; N=N)
end

function build_benchmark(
    ::Type{TensorContract4}, ::Type{T}, N, M
) where {T}
    return TensorContract4{T}(; N=N)
end

function initialize(b::TensorProjection3{T}; mod=cuNumeric) where {T}
    A = rand_array(mod, T, b.N, b.N, b.N)
    B = rand_array(mod, T, b.N, b.N)
    D = zeros_array(mod, T, b.N, b.N, b.N)
    GC.gc()
    return D, A, B
end

function initialize(b::TensorContract4{T}; mod=cuNumeric) where {T}
    X = rand_array(mod, T, b.N, b.N, b.N, b.N)
    Y = rand_array(mod, T, b.N, b.N, b.N, b.N)
    C = zeros_array(mod, T, b.N, b.N, b.N, b.N)
    GC.gc()
    return C, X, Y
end

function run!(::TensorProjection3, D, A, B)
    @tensor opt=true D[n, m, l] =
        A[i, j, k] * B[n, i] * B[m, j] * B[l, k]
    return D
end

function run!(::TensorContract4, C, X, Y)
    @tensor C[a, b, c, d] = X[a, i, c, j] * Y[i, b, j, d]
    return C
end

correctness_problem(b::TensorProjection3{T}) where {T} = TensorProjection3{T}(; N=min(b.N, 4))
correctness_problem(b::TensorContract4{T}) where {T} = TensorContract4{T}(; N=min(b.N, 4))
function correctness_atol_rtol(::AbstractTensorContraction, ::Type{T}) where {T}
    tol = T === Float32 ? 2.0f-4 : 1e-11
    return tol, tol
end

function benchmark_backend_label(
    ::AbstractTensorContraction, backend::String, default::String
)
    return backend == "cudajl" ? "TensorOperations.jl / cuTENSOR" : default
end

function benchmark_backend_save_as(
    ::AbstractTensorContraction, backend::String, default::String
)
    return backend == "cudajl" ? "tensoroperations_cuda" : default
end

register_benchmark("tensor_projection3", TensorProjection3)
register_benchmark("tensor_contract4", TensorContract4)
