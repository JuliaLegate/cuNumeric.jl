using Random
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

function _tensor_rand(mod, ::Type{T}, dims...) where {T}
    mod === CUDACore && return CUDACore.CuArray(rand(T, dims...))
    return mod.rand(T, dims...)
end

function initialize(b::TensorProjection3{T}; mod=cuNumeric) where {T}
    A = _tensor_rand(mod, T, b.N, b.N, b.N)
    B = _tensor_rand(mod, T, b.N, b.N)
    D = mod.zeros(T, b.N, b.N, b.N)
    GC.gc()
    return D, A, B
end

function initialize(b::TensorContract4{T}; mod=cuNumeric) where {T}
    X = _tensor_rand(mod, T, b.N, b.N, b.N, b.N)
    Y = _tensor_rand(mod, T, b.N, b.N, b.N, b.N)
    C = mod.zeros(T, b.N, b.N, b.N, b.N)
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

correctness_supported(::AbstractTensorContraction) = true

function _backend_array(mod, A)
    mod === cuNumeric && return NDArray(A)
    mod === CUDACore && return CUDACore.CuArray(A)
    return throw(ArgumentError("unsupported tensor contraction backend $mod"))
end

function _tensor_isapprox(actual, expected, ::Type{T}) where {T}
    tol = T === Float32 ? 2e-4 : 1e-11
    return isapprox(Array(actual), expected; atol=tol, rtol=tol)
end

function check_benchmark_correctness(
    b::TensorProjection3{T}, gs::GlobalSettings; mod=cuNumeric
) where {T}
    mod in (cuNumeric, CUDACore) || return "skipped"
    n = min(b.N, 4)
    rng = MersenneTwister(0x3b8a7c21)
    Ah = rand(rng, T, n, n, n)
    Bh = rand(rng, T, n, n)
    ref = zeros(T, n, n, n)
    @tensor opt=true ref[n, m, l] =
        Ah[i, j, k] * Bh[n, i] * Bh[m, j] * Bh[l, k]

    D = _backend_array(mod, zeros(T, n, n, n))
    A = _backend_array(mod, Ah)
    B = _backend_array(mod, Bh)
    run!(b, D, A, B)
    return _tensor_isapprox(D, ref, T) ? "pass" : "fail"
end

function check_benchmark_correctness(
    b::TensorContract4{T}, gs::GlobalSettings; mod=cuNumeric
) where {T}
    mod in (cuNumeric, CUDACore) || return "skipped"
    n = min(b.N, 4)
    rng = MersenneTwister(0xa3c97d42)
    Xh = rand(rng, T, n, n, n, n)
    Yh = rand(rng, T, n, n, n, n)
    ref = zeros(T, n, n, n, n)
    @tensor ref[a, b, c, d] = Xh[a, i, c, j] * Yh[i, b, j, d]

    C = _backend_array(mod, zeros(T, n, n, n, n))
    X = _backend_array(mod, Xh)
    Y = _backend_array(mod, Yh)
    run!(b, C, X, Y)
    return _tensor_isapprox(C, ref, T) ? "pass" : "fail"
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
