# Domain-aware test data generation

const DOMAIN_GENERATORS = Dict{Symbol,Function}(
    :greater_than_one => (T, N) -> abs.(rand(T, N)) .+ one(T),
    :unit_interval => (T, N) -> T(2) .* rand(T, N) .- one(T),
    :normal => (T, N) -> randn(T, N),
    :uniform => (T, N) -> rand(T, N),
    :positive => (T, N) -> abs.(rand(T, N)),
)

rtol(::Type{Float16}) = 1e-2
rtol(::Type{Float32}) = 1e-5
rtol(::Type{Float64}) = 1e-12
rtol(::Type{I}) where {I<:Integer} = rtol(float(I))
atol(::Type{Float16}) = 1e-3
atol(::Type{Float32}) = 1e-8
atol(::Type{Float64}) = 1e-15
atol(::Type{I}) where {I<:Integer} = atol(float(I))
rtol(::Type{Complex{T}}) where {T} = rtol(T)
atol(::Type{Complex{T}}) where {T} = atol(T)

is_same(arr1::NDArray, arr2::NDArray) = @allowscalar (arr1 == arr2)[1]
is_same(arr1::NDArray, arr2::Array) = @allowscalar (arr1 == arr2)[1]
is_same(arr1::Array, arr2::NDArray) = @allowscalar (arr1 == arr2)[1]
is_same(arr1::Array, arr2::Array) = (arr1 == arr2)

function my_rand(::Type{F}, dims...; L=F(-1000), R=F(1000)) where {F<:AbstractFloat}
    L .+ (R-L) .* rand(F, dims...)
end
function my_rand(::Type{I}, dims...; L=I(-255), R=I(255)) where {I<:Signed}
    L .+ floor.(I, (R - L + 1) .* rand(dims...))
end
my_rand(::Type{Bool}, dims...) = rand(Bool, dims...)

function make_julia_arrays(T, N, domain_key::Symbol; count::Int=1)
    generator = DOMAIN_GENERATORS[domain_key]

    julia_arrs_1D = [generator(T, N) for _ in 1:count]
    julia_arrs_2D = [reshape(generator(T, N), (isqrt(N), isqrt(N))) for _ in 1:count]

    return julia_arrs_1D..., julia_arrs_2D...
end

function make_cunumeric_arrays(julia_arrs_1D, julia_arrs_2D, T, N; count::Int=1)
    N_2D_dims = (isqrt(N), isqrt(N))

    cunumeric_arrs = [cuNumeric.zeros(T, N) for _ in 1:count]
    cunumeric_arrs_2D = [cuNumeric.zeros(T, N_2D_dims...) for _ in 1:count]

    @allowscalar begin
        for c in 1:count
            for i in 1:N
                cunumeric_arrs[c][i] = julia_arrs_1D[c][i]
            end
            for i in 1:N_2D_dims[1], j in 1:N_2D_dims[2]
                cunumeric_arrs_2D[c][i, j] = julia_arrs_2D[c][i, j]
            end
        end
    end

    return cunumeric_arrs..., cunumeric_arrs_2D...
end

function safe_isapprox(x, y, rtol, atol)
    # Handle NaN
    if isnan(x) && isnan(y)
        return true
    end

    # Handle Inf (must be same sign)
    if isinf(x) && isinf(y)
        return x === y
    end

    # Handle mixed finite/non-finite
    if isfinite(x) != isfinite(y)
        return false
    end

    return isapprox(x, y; rtol=rtol, atol=atol)
end

function safe_compare(x::AbstractArray{T}, y::NDArray{T}, rtol, atol) where {T}
    for CI in CartesianIndices(x)
        if !safe_isapprox(x[CI], y[Tuple(CI)...], rtol, atol)
            return false
        end
    end

    return true
end

function safe_compare(x::NDArray{T}, y::AbstractArray{T}, rtol, atol) where {T}
    return safe_compare(y, x)
end
